cd '/Users/pdealcan/Documents/github/CoE_Neto/code/accelProject/danceGenerator'
addpath('/Users/pdealcan/Documents/github/matlabTools/MocapToolbox/mocaptoolbox')
addpath('/Users/pdealcan/Documents/github/matlabTools/MIRtoolbox/MIRToolbox')

load mcdemodata

%Convert Brigitta's data to csv format
%%read data
% directoryIn = "/Users/pdealcan/Documents/github/data/CoE/accel/brigittaData/data/"
% directoryOut = "/Users/pdealcan/Documents/github/data/CoE/accel/brigittaData/convertedCSV/"
% 
% files = dir(directoryIn);
% for k=5:length(files)
%     a = load(strcat(directoryIn, files(k).name));
%     for l=1:length(a.pos)
%         data = a.pos(l).data;
%         nameSave = strcat(directoryOut, a.pos(l).participantName, "_", strrep(a.pos(l).sampleName,".wav",".csv"));
%         data = array2table(data);
%         writetable(data, nameSave);
%     end
% end

%Double checking marker positions for phone extraction
% a.pos(l)
% a.pos(l).markerName
% japar.markercolors='brrbbrbbbbbbbbbbbbbb'
% mcanimate(a.pos(l), japar)
%2, 3, 6 right knee, right hip and left hip

% markers = ["root", "lhip", "rhip", "belly", "lknee", "rknee", "spine", "lankle", "rankle", "chest", "ltoes", "rtoes", "neck", "linshoulder", "rinshoulder", "head",  "lshoulder", "rshoulder", "lelbow", "relbow", "lwrist", "rwrist", "lhand", "rhand"]

%%%%% CODE BELOW ALREADY TRANSFERED TO PYTHON
train = "test"
directoryIn = strcat("/Users/pdealcan/Documents/github/EDGEk/data/", train, "/motions_sliced_csv/") 
directoryOut = strcat("/Users/pdealcan/Documents/github/EDGEk/data/", train, "/positionsPhone/")

files = dir(directoryIn);
for k=3:length(files)
    fName = strcat(directoryIn, files(k).name);
    true = readtable(fName);
    true = table2array(true);
    
    df = dance1;
    df.nFrames = height(true);
    df.nMarkers = width(true)/3;
    df.freq = 30;

    df.data = true;

    phone = (mcgetmarker(df, 3).data + mcgetmarker(df, 6).data)/2;

    df.data = [df.data, phone];
    
    df.nMarkers=width(df.data)/3;

    %     par = mcinitanimpar
    %     par.markercolors = 'brrbbrbbbbbbbbbbbbbbbbbbr'
    %     mcanimate(df, par)
    
    %%%%%%% Calculate angles. 1=root; 2=up; 3=side
    xInd = 1;
    yInd = 3;
    zInd = 2;
    
    m1 = mcgetmarker(df, 25); %Phone root (root of phone)
    m2 = mcgetmarker(df, 3); %Right hip (upper phone)
    m3 = mcgetmarker(df, 6); %
     
    a1 = m1.data(:,xInd) - m2.data(:,xInd);
    a2 = m1.data(:,zInd) - m2.data(:,zInd);
    a3 = m2.data(:,zInd) - m3.data(:,zInd);
    
    dM1M2 = sqrt(sum((m1.data-m2.data).^2,2));
    dM2M3 = sqrt(sum((m2.data-m3.data).^2,2));
    % 
    pitch = asin(a1./dM1M2);
    yaw = asin(a2./dM1M2);
    roll = asin(a3./dM2M3);
    
    gyro = [pitch yaw roll];
    
    IMU = [mcgetmarker(df, 25).data gyro];
        
    df.data = IMU;
    df.nMarkers = width(df.data)/3;
    
    df = mctimeder(df, 2);
    phoneIMU = df.data;
        
    phoneIMU = array2table(phoneIMU);
    fNameOut = strcat(directoryOut, files(k).name)
    writetable(phoneIMU, fNameOut);
end


    %For some reason produces very innacurate results
%     phoneUp = (phone + mcgetmarker(trueD, 3).data)./2;
%     phoneUp = (phoneUp + phone)./2;
    
    %Phone laterals
%     middleHip = (mcgetmarker(trueD, 2).data + mcgetmarker(trueD, 3).data)/2;
%     phoneLateral = (middleHip + mcgetmarker(trueD, 6).data)/2;

%     entirePhone = [phone, phoneUp, phoneLateral];


