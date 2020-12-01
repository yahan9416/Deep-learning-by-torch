copy_number=readRDS("/mnt/Storage/home/hanya/project/Predict_essentiality_based_expression/Feature_collect/Feature_collect_result/Gene_itsel_across_expression_copy_mut.rds")
co_exp_ess=readRDS("/mnt/Storage/home/hanya/project/Predict_essentiality_based_expression/Feature_collect/Feature_collect_result/Gene_abs_top10_Co_Exp_Ess_across_expression.rds")
dim(copy_number)
essentiality=readRDS("/mnt/Storage/home/hanya/project/Predict_essentiality_based_expression/Feature_collect/Feature_collect_result/Crispr_across_cell_Line_normalized.rds")
co_exp_ess=co_exp_ess[,1:11]
copy_number=copy_number[,4]
copy_number[which(is.na(copy_number))]=0
co_exp_ess$copy_number=copy_number
co_exp_ess$essentiality=as.vector(essentiality)
library(randomForest)

setwd("/mnt/Storage/home/hanya/project/Predict_essentiality_based_expression/Feature_selection_linear_model/RandomForest_for_each_gene")
#train_model_pre<-NULL
test_model_pre<-NULL
test_model_pre_cor<<-NULL
model<-list()
one_gene_model<-function(gene){
  index=which(co_exp_ess[,2] == gene)
  data=as.matrix(co_exp_ess[index[1:385],3:13])
  ntree_fit<-randomForest(essentiality~.,data=data,mtry=1,ntree=500,importance=TRUE)

  dtest=as.matrix(co_exp_ess[index[386:551],3:12])
  y_pre=predict(ntree_fit,dtest)
  
  test_model_pre<<-cbind(test_model_pre,y_pre)
  model<<-c(model,gene<-list(ntree_fit))
  test_model_pre_cor<<-rbind(test_model_pre_cor,c(gene,unlist(cor.test(y_pre,co_exp_ess[index[386:551],13]))[c(3,4)]))
}
result=apply(matrix(as.vector(unique(co_exp_ess[,2]))),1,one_gene_model)

saveRDS(test_model_pre,file="top9_Abs_copy_comp_RF_model_test_predict.rds")
saveRDS(test_model_pre_cor,file="top9_Abs_copy_comp_RF_model_test_predict.rds")
saveRDS(model,file="top9_Abs_copy_comp_RF_model.rds")
