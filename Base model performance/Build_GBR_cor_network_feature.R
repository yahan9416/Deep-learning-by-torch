copy_number=readRDS("/mnt/Storage/home/hanya/project/Predict_essentiality_based_expression/Feature_collect/Feature_collect_result/Gene_itsel_across_expression_copy_mut.rds")
co_exp_ess=readRDS("/mnt/Storage/home/hanya/project/Predict_essentiality_based_expression/Feature_collect/Feature_collect_result/Gene_abs_top10_excep_self_Co_Exp_Ess_across_expression.rds")
expression=readRDS("/mnt/Storage/home/hanya/project/Predict_essentiality_based_expression/Feature_collect/Feature_collect_result/Expression_across_cell_Line_normalized.rds")
weight_neig_exp=readRDS("/mnt/Storage/home/hanya/project/Predict_essentiality_based_expression/Feature_collect/Feature_collect_result/Gene_top10_neigh_exc_self_weight_exp.rds")
colnames(weight_neig_exp)[3:12]=paste0("Neig_",colnames(weight_neig_exp)[3:12])
length(as.vector(unique(weight_neig_exp[,2])))
length(as.vector(unique(co_exp_ess[,2])))
gene_order=matrix(as.vector(unique(weight_neig_exp[,2])))
essentiality=readRDS("/mnt/Storage/home/hanya/project/Predict_essentiality_based_expression/Feature_collect/Feature_collect_result/Crispr_across_cell_Line_normalized.rds")
co_exp_ess$copy_number=copy_number[,4]
co_exp_ess$expression=copy_number[,3]
co_exp_ess=subset(co_exp_ess,Gene_id %in% gene_order)
co_exp_ess$Gene_id=as.vector(unlist(co_exp_ess$Gene_id))
co_exp_ess=cbind(co_exp_ess,weight_neig_exp[,3:12])
#which(co_exp_ess[,2] != weight_neig_exp[,2])
#length(levels(weight_neig_exp[,2]))
#length(levels(co_exp_ess[,2]))
essentiality=essentiality[,match(gene_order,colnames(essentiality))]
co_exp_ess$essentiality=as.vector(essentiality)
rm(weight_neig_exp)
rm(essentiality)
rm(copy_number)
library(gbm)
setwd("/mnt/Storage/home/hanya/project/Predict_essentiality_based_expression/Feature_selection_linear_model/GBR_regression_model_for_each_gene/Cor_feature_add_weight_neigh_exp")
#train_model_pre<-NULL
test_model_pre<-NULL
test_model_pre_cor<<-NULL
model<-list()
one_gene_model<-function(gene){
  index=which(co_exp_ess[,2] == gene)
  data=co_exp_ess[index[1:385],3:25]
  gene_model=gbm(essentiality ~ ., data=data, var.monotone = rep(0,22),n.trees = 500,cv.folds = 5, distribution = "gaussian",n.cores = 8)
  best.iter <- gbm.perf(gene_model, method = "cv")
  dtest=co_exp_ess[index[386:551],3:24]
  y_pre=predict(gene_model,dtest,n.trees=best.iter)
  
  test_model_pre<<-cbind(test_model_pre,y_pre)
  model<<-c(model,gene<-list(gene_model))
  test_model_pre_cor<<-rbind(test_model_pre_cor,c(gene,unlist(cor.test(y_pre,co_exp_ess[index[386:551],25]))[c(3,4)]))
}
result=apply(gene_order[1:10],1,one_gene_model)

saveRDS(test_model_pre,file="GBR_corr_network_feature_model_test_predict.rds")
saveRDS(test_model_pre_cor,file="GBR_corr_network_feature_model_test_predict_correlation_dis.rds")
saveRDS(model,file="GBR_corr_network_feature_model.rds")