setwd("/mnt/Storage/home/hanya/project/Predict_essentiality_based_expression/Feature_selection_linear_model/Co_exp_linear_regression")
library(xgboost)
negative_model=readRDS("posi_model.rds")
length(negative_model)
feature_name=negative_model[[1]][["feature_names"]]
#gene_name=readRDS("Feature_matrix_gene_order.rds")
#instead of each feature have been using into build model in the xgboost linear-regression model

extract_feature_importance<-function(x){
  feature_gain=xgb.importance( model = x)
  index=match(feature_name,feature_gain$Feature)
  return(as.vector(unlist(feature_gain$Gain[index])))
}
negative_result=matrix(unlist(lapply(negative_model,extract_feature_importance)),ncol=10,byrow=T)
colnames(negative_result)=feature_name
saveRDS(negative_result,file="Xgboost_Linear_feature_Gain_Co_exp.rds")