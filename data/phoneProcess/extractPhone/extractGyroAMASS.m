cd '/Users/pdealcan/Documents/github/CoE_Neto/code/accelProject/danceGenerator'
addpath('/Users/pdealcan/Documents/github/matlabTools/MocapToolbox/mocaptoolbox')
addpath('/Users/pdealcan/Documents/github/matlabTools/MIRtoolbox/MIRToolbox')

load mcdemodata

%Convert Brigitta's data to csv format
%%read data

directoryIn =  "/Users/pdealcan/Documents/github/data/CoE/accel/amass/DanceDBPoses/sliced/";
directoryOut = "/Users/pdealcan/Documents/github/data/CoE/accel/amass/DanceDBPoses/gyroPositions/";

files = dir(directoryIn);
for k=3:length(files)
    data = readtable(strcat(directoryIn, files(k).name));
    data = table2array(data);
    
    df = dance1;
    df.nMarkers = width(data)/3;
    df.nFrames = height(data);
    df.freq = 120;
    df.data = data;

    df = mcresample(df, 30);

    %Saving phone IMUs           
    phone = (mcgetmarker(df, 2).data + mcgetmarker(df, 3).data)/2;
        
        
    %%%%%%% Calculate angles. 1=root; 2=up; 3=side
    xInd = 1; %xzy also apply for brigitta's data
    yInd = 3;
    zInd = 2;
          
  %       hip = 2; knee = 3; lhip = 6
    m1 = mcgetmarker(df, 21); %Phone root (root of phone)
    m2 = mcgetmarker(df, 2); %Right hip (upper phone)
    m3 = mcgetmarker(df, 6); %left hip (side of phone)

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

    IMU = [phone gyro];
    
    df.data = IMU;
    df.nMarkers=width(df.data)/3;
    df.nFrames = height(df.data);

%     df = mctimeder(df, 2);

    IMU = df.data;
    IMU = array2table(IMU);     

    nameSave = strcat(directoryOut, files(k).name)

    writetable(IMU, nameSave);
end


