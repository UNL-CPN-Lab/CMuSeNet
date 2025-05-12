function [metadata, widebandSignal] = datagenWideband(SNRdB, fadingType)
      % Constant for this function
      RolloffFactor = 0.35;
      RaisedCosineFilterSpan = 10;
      Interpolation = 2;
      NarrowBandBWs = [1e5, 2e5, 5e5, 1e6, 2e6];
      WideBandBW = 20e6;
      MaxSignals = 10;
      %  Modulations = ["QPSK" "BPSK" "8-PSK" "8-QAM" "16-QAM" "2-FSK" ];
      Modulations = ["QPSK" "BPSK" "8-PSK" "8-QAM" "16-QAM", "GMSK", "2-FSK"];
      SamplingTime = 2/1000; % 2ms

      TxPowerRange = [0, 20];
      
      numberOfSignals = randi([1, MaxSignals], 1);
  
      signalBW = randsample(NarrowBandBWs, numberOfSignals, true);
      txPowers = randi(TxPowerRange, [numberOfSignals, 1]);

      
      minGap = 100e3; % 100kHz

      maxBW = max(signalBW);

      % Allocate a space for the frequencies
      freqOffsets = [];
      usedFreqs = [];
      % A mechanism to prevent it from being stuck if there are too many
      % wideband signals
      maxLoops = numberOfSignals * 10;
      % Generate non-overlapping frequencies
      for i = 1:numberOfSignals
        bw = signalBW(i);
        % Generate a random frequency offset within the limits
        while maxLoops > 0
            maxLoops = maxLoops -  1; % prevent it from handing
            freq = randi([-WideBandBW/2 + bw/2, WideBandBW/2 - bw/2]);
            % Check if the frequency space for the new signal is already occupied or
            % if the new signal is within minGap of an existing signal
            overlap = false;
            for j = 1:length(usedFreqs)
                existing_bw = signalBW(j);
                if abs(freq - usedFreqs(j)) < (bw + existing_bw)/2 + minGap
                    overlap = true;
                    break;
                end
            end
            if ~overlap
                % If not, add the frequency to the used frequencies and break the loop
                usedFreqs = [usedFreqs freq];
                freqOffsets = [freqOffsets freq];
                break
            end
            % If the frequency space is occupied or too close to another signal,
            % generate a new random frequency
        end

        if maxLoops <= 0
            numberOfSignals = length(freqOffsets);
            disp("Stopping because couldn't place signal");
            disp(signalBW);
            break;
        end

      end


      signals = [];
      metadata = [];

      lowestPowerSignal = min(txPowers);
      noisePower = min(txPowers) - SNRdB;

      
      for i = 1: numberOfSignals
          modulation = randsample(Modulations, 1);
          txPower = txPowers(i);
          bw = signalBW(i);
          % Should the divisor be 20 ?
          signal = datagenTransmitter( ...
              modulation, ...
              RolloffFactor, ...
              RaisedCosineFilterSpan, ...
              Interpolation, ...
              bw, ...
              SamplingTime...811
           );
          
          % Scale the signal
          signal = signal/sqrt(mean(abs(signal).^2));

          % Scale to correct power
          signal = 10^(txPower/20)*signal;

          pwr = 10*log10(mean(abs(signal).^2));
          
          
          if bw ~= maxBW
              signal = resample(signal, maxBW/1e5, bw/1e5);
          end
          signals = [signals signal];
          metadata = [metadata; struct("fc", freqOffsets(i), "bw", bw, "mod", modulation, "txPower", txPower, "noisePower", noisePower)];

      end
      mbc = comm.MultibandCombiner( ...
        InputSampleRate=maxBW, ...
        FrequencyOffsets=freqOffsets, ...
        OutputSampleRateSource="property", ...
        OutputSampleRate=WideBandBW ...
      );
      
      combinedsig = mbc(signals);
      % Channel configuration
      fd = 30; % Max Doppler shift in Hz
      Ts = 1/WideBandBW; % Sampling time
      chan = [];
        
      switch lower(fadingType)
          case 'awgn'
              % Just noise without fading
              widebandSignal = awgn(combinedsig, SNRdB, lowestPowerSignal);
    
          case 'rayleigh'
              rayleighChan = comm.RayleighChannel( ...
                  'SampleRate', WideBandBW, ...
                  'PathDelays', 0, ...
                  'AveragePathGains', 0, ...
                  'MaximumDopplerShift', 30 ...
              );
              fadedSignal = rayleighChan(combinedsig);
              widebandSignal = awgn(fadedSignal, SNRdB, lowestPowerSignal);  % Add AWGN
    
          case 'rician'
              ricianChan = comm.RicianChannel( ...
                  'SampleRate', WideBandBW, ...
                  'PathDelays', 0, ...
                  'AveragePathGains', 0, ...
                  'KFactor', 10, ...
                  'MaximumDopplerShift', 30 ...
              );
              fadedSignal = ricianChan(combinedsig);
              widebandSignal = awgn(fadedSignal, SNRdB, lowestPowerSignal);  % Add AWGN
    
          otherwise
              error('Unsupported fading type: %s', fadingType);
      end
end