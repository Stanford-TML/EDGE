source("/Users/pdealcan/Documents/github/doc_suomi/code/utils.R")

true = "/Users/pdealcan/Documents/github/data/CoE/accel/amass/DanceDBPoses/sliced_resampled/"
predicted = "/Users/pdealcan/Documents/github/data/CoE/accel/amass/DanceDBPoses/predictedBackup/"
predictedGyro = "/Users/pdealcan/Documents/github/data/CoE/accel/amass/DanceDBPoses/predictedGyro/"

files = list.files(predicted)
k = 1
for(k in 1:length(files)){
  f = files[k]
  fileTrue = paste(true, f, sep = "")

  filePredicted = paste(predicted, f, sep = "")
  #filePredicted2 = paste(predictedGyro, files[1], sep = "")

  trueD = fread(fileTrue)
  predictedD = fread(filePredicted)
  #predicted2 = fread(filePredicted2)

  markers = c("root", "lhip", "rhip", "belly", "lknee", "rknee", "spine", "lankle", "rankle", "chest", "ltoes", "rtoes", "neck", "linshoulder", "rinshoulder", "head",  "lshoulder", "rshoulder", "lelbow", "relbow", "lwrist", "rwrist", "lhand", "rhand")
  repeater = function(x){return(rep(x, 3))}

  cNames = unlist(lapply(markers, repeater))
  nRows = length(colnames(trueD))/3
  indexers = rep(c(1, 2, 3), 24)
  cNames = paste(cNames, indexers, sep = "")

  colnames(trueD) = cNames
  colnames(predictedD) = cNames

  trueD$condition = "true"
  predictedD$condition = "predicted"
  trueD$time = seq(1, length(trueD$condition))
  predictedD$time = seq(1, length(predictedD$condition))
  
  correlationC = cor(trueD$root3, predictedD$root3)

  df = bind_rows(trueD, predictedD)
  df %>%
    ggplot(aes(x = time, y = root3, color = condition))+
      facet_wrap(~condition, scale = "free")+
      geom_path()+
      labs(title=correlationC)
  ggsave(paste("/Users/pdealcan/Downloads/preds/", f, ".png", sep = ""))
}
