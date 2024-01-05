cd '/Users/pdealcan/Documents/github/CoE_Neto/code/accelProject/danceGenerator'
addpath('/Users/pdealcan/Documents/github/matlabTools/MocapToolbox/mocaptoolbox')
addpath('/Users/pdealcan/Documents/github/matlabTools/MIRtoolbox/MIRToolbox')

load mcdemodata

%Convert Brigitta's data to csv format
directoryIn =  "/Users/pdealcan/Documents/github/data/CoE/accel/amass/DanceDBPoses/sliced/";
directoryOut = "/Users/pdealcan/Documents/github/data/CoE/accel/amass/DanceDBPoses/sliced_resampled/";

files = dir(directoryIn);
for k=3:length(files)
    data = readtable(strcat(directoryIn, files(k).name));
    data = table2array(data);
    
    df = dance1;
    df.nMarkers = width(data)/3;
    df.nFrames = height(data);
    df.freq = 120;
    df.data = data;

    df = mcresample(df, 15);
    data = array2table(df.data);

    nameSave = strcat(directoryOut, files(k).name)

    writetable(data, nameSave);
end


