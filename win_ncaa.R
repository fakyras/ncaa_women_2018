library(dplyr)
library(xgboost)
library(lme4)

regresults <- read.csv("data/WRegularSeasonDetailedResults_PrelimData2018.csv")
results <- read.csv("data/WNCAATourneyDetailedResults_PrelimData2018.csv")
sub <- read.csv("data/WSampleSubmissionStage2.csv")
seeds <- read.csv("data/WNCAATourneySeeds.csv")

seeds$Seed = as.numeric(substring(seeds$Seed,2,4))


### Collect regular season results - double the data by swapping team positions

r1 = regresults[, c("Season", "DayNum", "WTeamID", "WScore", "LTeamID", "LScore", "NumOT", "WFGA", "WAst", "WBlk", "LFGA", "LAst", "LBlk")]
r2 = regresults[, c("Season", "DayNum", "LTeamID", "LScore", "WTeamID", "WScore", "NumOT", "LFGA", "LAst", "LBlk", "WFGA", "WAst", "WBlk")]
names(r1) = c("Season", "DayNum", "T1", "T1_Points", "T2", "T2_Points", "NumOT", "T1_fga", "T1_ast", "T1_blk", "T2_fga", "T2_ast", "T2_blk")
names(r2) = c("Season", "DayNum", "T1", "T1_Points", "T2", "T2_Points", "NumOT", "T1_fga", "T1_ast", "T1_blk", "T2_fga", "T2_ast", "T2_blk")
regular_season = rbind(r1, r2)


### Collect tourney results - double the data by swapping team positions

t1 = results[, c("Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore")] %>% mutate(ResultDiff = WScore - LScore)
t2 = results[, c("Season", "DayNum", "LTeamID", "WTeamID", "LScore", "WScore")] %>% mutate(ResultDiff = LScore - WScore)
names(t1) = c("Season", "DayNum", "T1", "T2", "T1_Points", "T2_Points", "ResultDiff")
names(t2) = c("Season", "DayNum", "T1", "T2", "T1_Points", "T2_Points", "ResultDiff")
tourney = rbind(t1, t2)


### Fit GLMM on regular season data (selected march madness teams only) - extract random effects for each team

march_teams = select(seeds, Season, Team = TeamID)
X =  regular_season %>% 
  inner_join(march_teams, by = c("Season" = "Season", "T1" = "Team")) %>% 
  inner_join(march_teams, by = c("Season" = "Season", "T2" = "Team")) %>% 
  select(Season, T1, T2, T1_Points, T2_Points, NumOT) %>% distinct()
X$T1 = as.factor(X$T1)
X$T2 = as.factor(X$T2)

quality = list()
for (season in unique(X$Season)) {
  glmm = glmer(I(T1_Points > T2_Points) ~  (1 | T1) + (1 | T2), data = X[X$Season == season & X$NumOT == 0, ], family = binomial) 
  random_effects = ranef(glmm)$T1
  quality[[season]] = data.frame(Season = season, Team_Id = as.numeric(row.names(random_effects)), quality = exp(random_effects[,"(Intercept)"]))
}
quality = do.call(rbind, quality)


### Regular season statistics

season_summary = 
  regular_season %>%
  mutate(win14days = ifelse(DayNum > 118 & T1_Points > T2_Points, 1, 0),
         last14days = ifelse(DayNum > 118, 1, 0)) %>% 
  group_by(Season, T1) %>%
  summarize(
    WinRatio14d = sum(win14days) / sum(last14days),
    PointsMean = mean(T1_Points),
    PointsMedian = median(T1_Points),
    PointsDiffMean = mean(T1_Points - T2_Points),
    FgaMean = mean(T1_fga), 
    FgaMedian = median(T1_fga),
    FgaMin = min(T1_fga), 
    FgaMax = max(T1_fga), 
    AstMean = mean(T1_ast), 
    BlkMean = mean(T1_blk), 
    OppFgaMean = mean(T2_fga), 
    OppFgaMin = min(T2_fga)  
  )

season_summary_X1 = season_summary
season_summary_X2 = season_summary
names(season_summary_X1) = c("Season", "T1", paste0("X1_",names(season_summary_X1)[-c(1,2)]))
names(season_summary_X2) = c("Season", "T2", paste0("X2_",names(season_summary_X2)[-c(1,2)]))


### Combine all features into a data frame

data_matrix =
  tourney %>% 
  left_join(season_summary_X1, by = c("Season", "T1")) %>% 
  left_join(season_summary_X2, by = c("Season", "T2")) %>%
  left_join(select(seeds, Season, T1 = TeamID, Seed1 = Seed), by = c("Season", "T1")) %>% 
  left_join(select(seeds, Season, T2 = TeamID, Seed2 = Seed), by = c("Season", "T2")) %>% 
  mutate(SeedDiff = Seed1 - Seed2) %>%
  left_join(select(quality, Season, T1 = Team_Id, quality_march_T1 = quality), by = c("Season", "T1")) %>%
  left_join(select(quality, Season, T2 = Team_Id, quality_march_T2 = quality), by = c("Season", "T2"))


### Prepare xgboost 

features = setdiff(names(data_matrix), c("Season", "DayNum", "T1", "T2", "T1_Points", "T2_Points", "ResultDiff"))
dtrain = xgb.DMatrix(as.matrix(data_matrix[, features]), label = data_matrix$ResultDiff)

cauchyobj <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  c <- 5000 
  x <-  preds-labels
  grad <- x / (x^2/c^2+1)
  hess <- -c^2*(x^2-c^2)/(x^2+c^2)^2
  return(list(grad = grad, hess = hess))
}

xgb_parameters = 
  list(objective = cauchyobj, 
       eval_metric = "mae",
       booster = "gbtree", 
       eta = 0.02,
       subsample = 0.35,
       colsample_bytree = 0.7,
       num_parallel_tree = 10,
       min_child_weight = 40,
       gamma = 10,
       max_depth = 3)

N = nrow(data_matrix)
fold5list = c(
  rep( 1, floor(N/5) ),
  rep( 2, floor(N/5) ),
  rep( 3, floor(N/5) ),
  rep( 4, floor(N/5) ),
  rep( 5, N - 4*floor(N/5) )
)


### Build cross-validation model, repeated 10-times

iteration_count = c()
smooth_model = list()

for (i in 1:10) {
  
  ### Resample fold split
  set.seed(i)
  folds = list()  
  fold_list = sample(fold5list)
  for (k in 1:5) folds[[k]] = which(fold_list == k)
  
  set.seed(120)
  xgb_cv = 
    xgb.cv(
      params = xgb_parameters,
      data = dtrain,
      nrounds = 3000,
      verbose = 0,
      nthread = 12,
      folds = folds,
      early_stopping_rounds = 25,
      maximize = FALSE,
      prediction = TRUE
    )
  iteration_count = c(iteration_count, xgb_cv$best_iteration)
  
  ### Fit a smoothed GAM model on predicted result point differential to get probabilities
  smooth_model[[i]] = smooth.spline(x = xgb_cv$pred, y = ifelse(data_matrix$ResultDiff > 0, 1, 0))
  
}


### Build submission models

submission_model = list()

for (i in 1:10) {
  set.seed(i)
  submission_model[[i]] = 
    xgb.train(
      params = xgb_parameters,
      data = dtrain,
      nrounds = round(iteration_count[i]*1.05),
      verbose = 0,
      nthread = 12,
      maximize = FALSE,
      prediction = TRUE
    )
}


### Run predictions

sub$Season = 2018
sub$T1 = as.numeric(substring(sub$ID,6,9))
sub$T2 = as.numeric(substring(sub$ID,11,14))

Z = sub %>% 
  left_join(season_summary_X1, by = c("Season", "T1")) %>% 
  left_join(season_summary_X2, by = c("Season", "T2")) %>%
  left_join(select(seeds, Season, T1 = TeamID, Seed1 = Seed), by = c("Season", "T1")) %>% 
  left_join(select(seeds, Season, T2 = TeamID, Seed2 = Seed), by = c("Season", "T2")) %>% 
  mutate(SeedDiff = Seed1 - Seed2) %>%
  left_join(select(quality, Season, T1 = Team_Id, quality_march_T1 = quality), by = c("Season", "T1")) %>%
  left_join(select(quality, Season, T2 = Team_Id, quality_march_T2 = quality), by = c("Season", "T2"))

dtest = xgb.DMatrix(as.matrix(Z[, features]))

probs = list()
for (i in 1:10) {
  preds = predict(submission_model[[i]], dtest)
  probs[[i]] = predict(smooth_model[[i]], preds)$y
}
Z$Pred = Reduce("+", probs) / 10

### Better be safe than sorry
Z$Pred[Z$Pred <= 0.025] = 0.025
Z$Pred[Z$Pred >= 0.975] = 0.975

### Anomaly event happened only once before - be brave
Z$Pred[Z$Seed1 == 16 & Z$Seed2 == 1] = 0
Z$Pred[Z$Seed1 == 15 & Z$Seed2 == 2] = 0
Z$Pred[Z$Seed1 == 14 & Z$Seed2 == 3] = 0
Z$Pred[Z$Seed1 == 13 & Z$Seed2 == 4] = 0
Z$Pred[Z$Seed1 == 1 & Z$Seed2 == 16] = 1
Z$Pred[Z$Seed1 == 2 & Z$Seed2 == 15] = 1
Z$Pred[Z$Seed1 == 3 & Z$Seed2 == 14] = 1
Z$Pred[Z$Seed1 == 4 & Z$Seed2 == 13] = 1

write.csv(select(Z, ID, Pred), "sub.csv", row.names = FALSE)

