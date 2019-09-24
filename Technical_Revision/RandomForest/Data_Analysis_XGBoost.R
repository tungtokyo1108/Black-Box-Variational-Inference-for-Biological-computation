library(rsample)
library(dplyr)
library(gbm)
library(xgboost)
library(caret)
library(pdp)
library(ggplot2)
library(lime)
library(DiagrammeR)


mydata <- read.csv("feature_full.csv")
cat_data <- read.csv("data_raw.csv")
mydata$Categories <- as.factor(cat_data$category_label)
attach(cat_data)
attach(mydata)

# Create training data (70%) and test (30%)
mydata_split <- initial_split(mydata, prop = .7)
mydata_train <- training(mydata_split)
mydata_test <- testing(mydata_split)
attach(mydata_train)
attach(mydata_test)
xgb_data_train <- as.matrix(select(mydata_train, -c(Categories)))
xgb_data_test <- as.matrix(select(mydata_test, -c(Categories)))

dtrain <- xgb.DMatrix(data = xgb_data_train, label = as.numeric(mydata_train$Categories) - 1)
dtest <- xgb.DMatrix(data = xgb_data_test, label = as.numeric(mydata_test$Categories) -1)
watchlist <- list(train = dtrain, eval = dtest)
num_class <- length(levels(mydata_train$Categories))
nrounds <- 20

# Train model 
hyper_grid <- expand.grid(
  eta = c(.01, .05, .1, .3),
  max_depth = c(1, 3, 5, 7, 9),
  # max_depth = seq(1, 10, by = 1),
  min_child_weight = c(1, 3, 5, 7, 9),
  #min_child_weight = seq(1, 10, by = 1),
  subsample = c(.65, .8, 1),
  colsample_bytree = c(.8, .9, 1),
  optimal_trees = 0,
  min_merror_mean = 0
)

for (i in 1:nrow(hyper_grid)) {
  params <- list(
    eta = hyper_grid$eta[i],
    max_depth = hyper_grid$max_depth[i],
    min_child_weight = hyper_grid$min_child_weight[i],
    subsample = hyper_grid$subsample[i],
    colsample_bytree = hyper_grid$colsample_bytree[i]
  )
  
  set.seed(123)
  
  xgb.tune <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = nrounds,
    nthread = 8,
    nfold = 10,
    objective = "multi:softprob",
    num_class = num_class,
    verbose = 0,
    early_stopping_rounds = 10
  )
  
  hyper_grid$optimal_trees[i] <- which.min(xgb.tune$evaluation_log$test_merror_mean)
  hyper_grid$min_merror_mean[i] <- min(xgb.tune$evaluation_log$test_merror_mean)
}

hyper_grid %>%
  dplyr::arrange(min_merror_mean) %>%
  head(10)

# Final model 
params_final <- list(
  eta = 0.3,
  max_depth = 3,
  min_child_weight = 3,
  subsample = 0.65,
  colsample_bytree = 0.8
)

bstDMatrix <- xgboost(data = dtrain, params = params_final, 
                      nthread = 8, nrounds = 10, 
                      objective = "multi:softprob", num_class = num_class)

# Explain model 
xgb.ggplot.deepness(bstDMatrix)
xgb.plot.deepness(bstDMatrix, which = "max.depth", pch=16, col=rgb(0,0,1,0.3), cex=2)


pdf("plot/XGB_Importance_feature.pdf")
# All classes
importance_matrix <- xgb.importance(colnames(xgb_data_train), model = bstDMatrix)
xgb.plot.importance(importance_matrix, measure = "Gain", 
                    rel_to_first = TRUE, xlab = "Relative importance")
xgb.ggplot.importance(importance_matrix, rel_to_first = TRUE, measure = "Gain")

# inspect importances separately for each class
importance_cat0 <- xgb.importance(model = bstDMatrix, trees = seq(from=0, by=num_class,
                                            length.out = nrounds))
xgb.plot.importance(importance_cat0, rel_to_first = TRUE, measure = "Gain", 
                    main = "Importance Features for Cat_0", xlab = "Relative importance")
importance_cat1 <- xgb.importance(model = bstDMatrix, trees = seq(from=1, by=num_class, 
                                            length.out = nrounds))
xgb.plot.importance(importance_cat1, rel_to_first = TRUE, measure = "Gain",
                    main = "Importance Features for Cat_1", xlab = "Relative importance")
importance_cat2 <- xgb.importance(model = bstDMatrix, trees = seq(from=2, by=num_class, 
                                                                  length.out = nrounds))
xgb.plot.importance(importance_cat2, rel_to_first = TRUE, measure = "Gain",
                    main = "Importance Features for Cat_2", xlab = "Relative importance")
importance_cat3 <- xgb.importance(model = bstDMatrix, trees = seq(from=3, by=num_class, 
                                                                  length.out = nrounds))
xgb.plot.importance(importance_cat3, rel_to_first = TRUE, measure = "Gain",
                    main = "Importance Features for Cat_3", xlab = "Relative importance")
importance_cat4 <- xgb.importance(model = bstDMatrix, trees = seq(from=4, by=num_class, 
                                                                  length.out = nrounds))
xgb.plot.importance(importance_cat4, rel_to_first = TRUE, measure = "Gain",
                    main = "Importance Features for Cat_4", xlab = "Relative importance")
dev.off()

# Plot multi-tree 
gr <- xgb.plot.multi.trees(model = bstDMatrix, features_keep = 5, render = FALSE)
export_graph(gr, "tree.pdf", width = 1500, height = 1000)

# Plot each tree with node ID
gr <- xgb.plot.tree(model = bstDMatrix, trees = 0:2, show_node_id = TRUE, render = FALSE)
export_graph(gr, "plot/XGB_tree.pdf", width = 1500, height = 1500)

# Plot shap for each category
pdf("plot/XGB_shap.pdf")
xgb.plot.shap(xgb_data_train, model = bstDMatrix, trees = seq(from=0, by=num_class, length.out = nrounds),
              target_class = 0, top_n = 6, n_col = 2, col = rgb(0, 0, 1, 0.5), 
              main = "Cat_0", pch = 16, pch_NA = 17)
xgb.plot.shap(xgb_data_train, model = bstDMatrix, trees = seq(from=1, by=num_class, length.out = nrounds),
              target_class = 1, top_n = 6, n_col = 2, col = rgb(0, 0, 1, 0.5), 
              main = "Cat_1", pch = 16, pch_NA = 17)
xgb.plot.shap(xgb_data_train, model = bstDMatrix, trees = seq(from=2, by=num_class, length.out = nrounds),
              target_class = 2, top_n = 6, n_col = 2, col = rgb(0, 0, 1, 0.5), 
              main = "Cat_2", pch = 16, pch_NA = 17)
xgb.plot.shap(xgb_data_train, model = bstDMatrix, trees = seq(from=3, by=num_class, length.out = nrounds),
              target_class = 3, top_n = 6, n_col = 2, col = rgb(0, 0, 1, 0.5), 
              main = "Cat_3", pch = 16, pch_NA = 17)
xgb.plot.shap(xgb_data_train, model = bstDMatrix, trees = seq(from=4, by=num_class, length.out = nrounds),
              target_class = 4, top_n = 6, n_col = 2, col = rgb(0, 0, 1, 0.5), 
              main = "Cat_4", pch = 16, pch_NA = 17)
dev.off()

# Local explain random forest
explainer_xgb <- lime(data.frame(xgb_data_train), bstDMatrix)
class(explainer_xgb)
summary(explainer_xgb)
index <- 1:4
local_obs <- xgb_data_test[index,]

pdf("plot/XGB_Lime.pdf")
explaination <- explain(x = data.frame(local_obs), explainer_xgb, n_features = 10, labels = 1)
plot_features(explaination)
explaination <- explain(x = data.frame(local_obs), explainer_xgb, n_features = 10, labels = 2)
plot_features(explaination)
explaination <- explain(x = data.frame(local_obs), explainer_xgb, n_features = 10, labels = 3)
plot_features(explaination)
explaination <- explain(x = data.frame(local_obs), explainer_xgb, n_features = 10, labels = 4)
plot_features(explaination)
explaination <- explain(x = data.frame(local_obs), explainer_xgb, n_features = 10, labels = 5)
plot_features(explaination)
dev.off()

predDMatrix <- predict(bstDMatrix, newdata = dtest, reshape = T)
xgb.pred <- as.data.frame(predDMatrix)
colnames(xgb.pred) <- levels(mydata_test$Categories)
xgb.pred$prediction <- apply(xgb.pred,1,function(x) colnames(xgb.pred)[which.max(x)])
xgb.pred$label <- levels(mydata_test$Categories)[as.numeric(mydata_test$Categories)]
confusionMatrix(as.factor(xgb.pred$prediction), as.factor(mydata_test$Categories), 
                mode = "everything")
