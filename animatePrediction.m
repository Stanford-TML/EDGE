cd '/Users/pdealcan/Documents/github/CoE_Neto/code/accelProject/danceGenerator'
addpath('/Users/pdealcan/Documents/github/matlabTools/MocapToolbox/mocaptoolbox')
addpath('/Users/pdealcan/Documents/github/matlabTools/MIRtoolbox/MIRToolbox')

load mcdemodata

%%read data
% directoryIn = "/Users/pdealcan/Documents/github/EDGEk/generatedDance/pairGenerated/"
% files = dir(directoryIn);

dirTruth = "/Users/pdealcan/Documents/github/data/CoE/accel/amass/DanceDBPoses/sliced/"
files = dir("/Users/pdealcan/Documents/github/data/CoE/accel/amass/DanceDBPoses/predicted/");
randIndex = randi([3 length(files)]);


true = strcat(dirTruth, files(randIndex).name);
pred = strcat(files(randIndex).folder, "/", files(randIndex).name);

true = readtable(true);
pred = readtable(pred);

true = table2array(true);
pred = table2array(pred);

% markers = ["root", "lhip", "rhip", "belly", "lknee", "rknee", "spine", "lankle", "rankle", "chest", "ltoes", "rtoes", "neck", "linshoulder", "rinshoulder", "head",  "lshoulder", "rshoulder", "lelbow", "relbow", "lwrist", "rwrist", "lhand", "rhand"]



%Adding to matlab object
df = dance1;
df.nFrames = height(true);
df.nMarkers = width(true)/3;
df.freq = 15;

trueD = df;
predE = df;

trueD.data = true;
trueD.freq = 120;
predE.data = pred;
trueD.nMarkers=width(trueD.data)/3
predE.nMarkers=width(predE.data)/3

%Resampling to the same as AIST++
trueD = mcresample(trueD, 15);

%Parameters for aist dataset
par = mcinitanimpar
par.msize = 8
par.output = "mp4";
par.videoformat = 'mp4'
par.conn = [1 2; 1 3; 1 4; 3 6; 2 5; 3 6; 4 7; 5 8; 6 9; 9 12; 8 11; 7 10; 10 13; 13 16; 10 14; 10 15; 14 17; 15 18; 18 20; 17 19; 20 22; 19 21; 21 23]
par.markercolors='bbbbbbbbbbbbbbbbbbbbrbbb'


par2 = mcinitanimpar
par2.msize = 8
par2.output = "mp4";
par2.videoformat = 'mp4'
par2.conn = [1 2; 1 3; 1 4; 3 6; 2 5; 3 6; 4 7; 5 8; 6 9; 9 12; 8 11; 7 10; 10 13; 13 16; 10 14; 10 15; 14 17; 15 18; 18 20; 17 19; 20 22; 19 21; 21 23]
par2.markercolors='rrrrrrrrrrrrrrrrrrrrbrrr'


trueD = mccenter(trueD);
predE = mccenter(predE);

[all, allparams] = mcmerge(trueD, mctranslate(predE, [2 0 0]), par, par2);
% end

% mcanimate(all, allparams)
% trueD.data = [trueD.data phoneRoot]
% trueD.nMarkers=width(trueD.data)/3

allparams.videoformat = "mp4"
mcanimate(all, allparams);



