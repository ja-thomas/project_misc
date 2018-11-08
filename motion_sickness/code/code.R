library(readr)
library(dplyr)
library(ggplot2)
library(tidyr)
library(mlr)

data = read_csv2("data/NewMotionSicknessLog_merged.csv")


data = data %>%
  mutate(
    HeadVelAbs = sqrt(HeadVelX_filtered^2 + HeadVelY_filtered^2 + HeadVelZ_filtered^2),
    TorsoVelAbs = sqrt(TorsoVelX_filtered^2 + TorsoVelY_filtered^2 + TorsoVelZ_filtered^2),
    HeadRotationVelAbs = sqrt(HeadRotationVelX_filtered^2 + HeadRotationVelY_filtered^2 + HeadRotationVelZ_filtered^2)) %>%
  mutate_at(vars(matches("Abs")), funs(mean, var), na.rm = TRUE)

fda_feats = c(
  "HeadPosX", "HeadPosY", "HeadPosZ",
  "HeadVelX_filtered", "HeadVelY_filtered", "HeadVelZ_filtered",
  "TorsoPosX", "TorsoPosY", "TorsoPosZ",
  "TorsoVelX_filtered", "TorsoVelY_filtered", "TorsoVelZ_filtered",
  "HeadRotationX", "HeadRotationY", "HeadRotationZ",
  "HeadRotationVelX_filtered", "HeadRotationVelY_filtered", "HeadRotationVelZ_filtered",
  "HeadVelAbs", "TorsoVelAbs", "HeadRotationVelAbs")


getFDFeats = function(feats, cols) {
  res = lapply(feats, function(x) which(grepl(x, cols)))
  names(res) = feats
  res
}


data2 = data %>%
  filter(Phase == "post", Task != "Liegend") %>%
  select(BL_SSQ_Total, sex, age, height, weight, Participant, TrialNo, Time,
    one_of(fda_feats), matches("mean"), matches("var")) %>%
  mutate(Time = round(Time - 80, 2), sex = as.factor(sex)) %>%
  group_by(Participant, TrialNo) %>%
  gather(var, val, one_of(fda_feats)) %>%
  group_by(Time) %>%
  unite(VarT, var, Time, sep="_") %>%
  spread(VarT, val, fill=0)

data2 = data2 %>%
  makeFunctionalData(fd.features = getFDFeats(feats = fda_feats, cols = colnames(.))) %>%
  select(-TrialNo) %>%
  mutate(Participant = as.factor(Participant))


task = makeRegrTask(data = select(data2, -Participant), target = "BL_SSQ_Total", blocking = data2$Participant)


feat.methods = list(all = extractFDAFourier(), all = extractFDAWavelets(), all = extractFDAMultiResFeatures())

lrn = makeLearner("regr.ranger", importance = "impurity")
wrapped.lrn = makeExtractFDAFeatsWrapper(lrn,  feat.methods = feat.methods)

cv = makeResampleDesc("CV", iters = 23)

res = benchmark(list("regr.featureless", wrapped.lrn), task, cv, mae)

mod = train(wrapped.lrn, task)
getFeatureImportance(mod)
