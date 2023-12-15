cd '/Users/pdealcan/Documents/github/EDGEk/generatedDance/custom/'
addpath('/Users/pdealcan/Documents/github/matlabTools/MocapToolbox/mocaptoolbox')
addpath('/Users/pdealcan/Documents/github/matlabTools/MIRtoolbox/MIRToolbox')

load mcdemodata

%%read data
files = dir();
f = 'Accelerometer.csv'

accel = readtable(f);

timeTotal = max(accel.seconds_elapsed)
nSamples = height(accel.seconds_elapsed)
freq = (timeTotal/3810)*1000

accel = table2array(accel);
accel = accel(:,3:end)

df = dance1;
df.nFrames = height(accel);
df.nMarkers = width(accel)/3;
df.freq = freq;
df.data = accel;

df = mcresample(df, 15);
df = mctrim(df, 23, 33);

df.data = df.data(2:end,:);
pro = array2table(df.data);
writetable(pro, "/Users/pdealcan/Documents/github/EDGEk/generatedDance/custom/accelerometerProcessed.csv")
