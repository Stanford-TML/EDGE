cd '/Users/pdealcan/Documents/github/CoE_Neto/code/accelProject/danceGenerator'
addpath('/Users/pdealcan/Documents/github/matlabTools/MocapToolbox/mocaptoolbox')
addpath('/Users/pdealcan/Documents/github/matlabTools/MIRtoolbox/MIRToolbox')

load mcdemodata

gyro = "false";

%Convert Brigitta's data to csv format
%%read data
directoryIn = "/Users/pdealcan/Documents/github/data/CoE/accel/brigittaData/data/";
directoryOut = strcat("/Users/pdealcan/Documents/github/data/CoE/accel/brigittaData/convertedCSV/");
directoryOut2 = strcat("/Users/pdealcan/Documents/github/data/CoE/accel/brigittaData/accel/phoneIMU/");

files = dir(directoryIn);
for k=5:length(files)
    a = load(strcat(directoryIn, files(k).name));
    for l=1:length(a.pos)
        df = a.pos(l);
        df = mcresample(df, 15);
        data = df.data;
        nameSave = strcat(directoryOut, a.pos(l).participantName, "_", strrep(a.pos(l).sampleName,".wav",".csv"));
        data = array2table(data);
        
        %Saving full body movement
        writetable(data, nameSave); 

        %Saving phone IMUs           
        phone = (mcgetmarker(df, 2).data + mcgetmarker(df, 3).data)/2;
        
        df.data = [df.data, phone];
        df.nMarkers=width(df.data)/3;
        df.nFrames = height(df.data);
        df.markerName = [df.markerName {'phoneRoot'}];
        
        IMU = [mcgetmarker(df, 21).data];
        
        df.data = IMU;
        df.nMarkers = width(df.data)/3;

        df = mctimeder(df, 2);

        phoneIMU = df.data;
      
        phoneIMU = array2table(phoneIMU);
        nameSave = strcat(directoryOut2, a.pos(l).participantName, "_", strrep(a.pos(l).sampleName,".wav",".csv"))
        writetable(phoneIMU, nameSave);
        
    end
end

%Double checking marker positions for phone extraction
% a.pos(l)
% a.pos(l).markerName
% japar.markercolors='brrbbrbbbbbbbbbbbbbb'
% mcanimate(a.pos(l), japar)
%2, 3, 6 right knee, right hip and left hip

% markers = ["root", "lhip", "rhip", "belly", "lknee", "rknee", "spine", "lankle", "rankle", "chest", "ltoes", "rtoes", "neck", "linshoulder", "rinshoulder", "head",  "lshoulder", "rshoulder", "lelbow", "relbow", "lwrist", "rwrist", "lhand", "rhand"]
