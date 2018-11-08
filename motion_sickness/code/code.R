library(readr)
library(dplyr)
library(ggplot2)
library(tidyr)
library(mlr)

data = read_csv2("data/NewMotionSicknessLog_merged_v2.csv")


data = data %>%
  mutate(
    HeadVelAbs = sqrt(HeadVelX^2 + HeadVelY^2 + HeadVelZ^2),
    TorsoVelAbs = sqrt(TorsoVelX^2 + TorsoVelY^2 + TorsoVelZ^2),
    HeadRotationVelAbs = sqrt(HeadRotationVelX^2 + HeadRotationVelY^2 + HeadRotationVelZ^2)) %>%
  mutate_at(vars(matches("Abs")), funs(mean, var), na.rm = TRUE)


fda_feats = c(
  "HeadPosX", "HeadPosY", "HeadPosZ",
  "HeadVelX", "HeadVelY", "HeadVelZ",
  "TorsoPosX", "TorsoPosY", "TorsoPosZ",
  "TorsoVelX", "TorsoVelY", "TorsoVelZ",
  "HeadRotationX", "HeadRotationY", "HeadRotationZ",
  "HeadRotationVelX", "HeadRotationVelY", "HeadRotationVelZ",
  "HeadVelAbs", "TorsoVelAbs", "HeadRotationVelAbs")


getFDFeats = function(feats, cols) {
  res = lapply(feats, function(x) which(grepl(x, cols)))
  names(res) = feats
  res
}


data2 = data %>%
  filter(Phase == "post", Task != "Liegend") %>%
  select(SSQ_Total_diff, sex, age, height, weight, Participant, TrialNo, Time,
    one_of(fda_feats), matches("mean"), matches("var")) %>%
  mutate(Time = round(Time - 80, 2), sex = as.factor(sex)) %>%
  group_by(Participant, TrialNo) %>%
  gather(var, val, one_of(fda_feats)) %>%
  group_by(Time) %>%
  unite(VarT, var, Time, sep="_") %>%
  spread(VarT, val, fill=0)

data2 = data2 %>%
  makeFunctionalData(fd.features = getFDFeats(feats = fda_feats, cols = colnames(.))) %>%
#  select(-TrialNo) %>%
  mutate(Participant = as.factor(Participant))


task = makeRegrTask(data = select(data2, -Participant), target = "SSQ_Total_diff", blocking = data2$Participant)


feat.methods = list(all = extractFDAFourier(), all = extractFDAWavelets(), all = extractFDAMultiResFeatures())

lrn = makeLearner("regr.ranger", importance = "impurity")
wrapped.lrn = makeExtractFDAFeatsWrapper(lrn,  feat.methods = feat.methods)

cv = makeResampleDesc("CV", iters = 23)

res = benchmark(list("regr.featureless", wrapped.lrn), task, cv, mae)

mod = train(wrapped.lrn, task)
getFeatureImportance(mod)
