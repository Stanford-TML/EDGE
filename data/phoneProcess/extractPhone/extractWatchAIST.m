cd '/Users/pdealcan/Documents/github/CoE_Neto/code/accelProject/danceGenerator'
addpath('/Users/pdealcan/Documents/github/matlabTools/MocapToolbox/mocaptoolbox')
addpath('/Users/pdealcan/Documents/github/matlabTools/MIRtoolbox/MIRToolbox')

load mcdemodata

%2, 3, 6 right knee, right hip and left hip
% markers = ["root", "lhip", "rhip", "belly", "lknee", "rknee", "spine", "lankle", "rankle", "chest", "ltoes", "rtoes", "neck", "linshoulder", "rinshoulder", "head",  "lshoulder", "rshoulder", "lelbow", "relbow", "lwrist", "rwrist", "lhand", "rhand"]
%%%%% Extracting phone IMUs from AIST++. For training and testing
traintest = ["test", "train"];
fType = ["accel"];
for l=1:length(fType)
  for j=1:length(traintest)
      directoryIn = strcat("/Users/pdealcan/Documents/github/EDGEk/data/", fType(l), "/", traintest(j), "/motions_sliced_csv/");
      directoryOut = strcat("/Users/pdealcan/Documents/github/EDGEk/data/", fType(l), "/", traintest(j), "/positionsWatch/");

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
          watch = mcgetmarker(df, 21).data;

          df.data = [df.data, phone, watch];

          df.nMarkers = width(df.data)/3;
   
          df = mctimeder(df, 2);

          phoneIMU = df.data;

          phoneIMU = array2table(phoneIMU);
          fNameOut = strcat(directoryOut, files(k).name)
          writetable(phoneIMU, fNameOut);
      end
  end
end
