path = "../diff-snr-matlab-simulated-data";

for snr = -20:2:10
    disp(snr);
    mkdir(sprintf("%s/%d/", path, snr));
    for i = 1:5000
        name = string(i);
        channelType = 'awgn'; %Supported channel type: awgn, rician (Flat), rayleigh (Flat)
        [meta, data] = datagenWideband(snr, channelType);
        split = reshape([real(data) imag(data)].', 1, []);
        
        % Save data file
        mkdir(sprintf("%s/%d/%s", path, snr, name));
        
        
        datafile = fopen(sprintf("%s/%d/%s/data.dat", path, snr, name), 'w');
        fwrite(datafile, split, 'double');
        fclose(datafile);
        
        % Save meta file
        metafile = fopen(sprintf("%s/%d/%s/meta-data.json", path, snr, name), 'w');
        fprintf(metafile, jsonencode(meta));
        fclose(metafile);
        
        
        disp(name);
    end
end