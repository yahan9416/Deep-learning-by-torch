library(lars)
copy_number=readRDS("/mnt/Storage/home/hanya/project/Predict_essentiality_based_expression/Feature_collect/Feature_collect_result/Gene_itsel_across_expression_copy_mut.rds")
co_exp_ess=readRDS("/mnt/Storage/home/hanya/project/Predict_essentiality_based_expression/Feature_collect/Feature_collect_result/Gene_abs_top10_excep_self_Co_Exp_Ess_across_expression.rds")
expression=readRDS("/mnt/Storage/home/hanya/project/Predict_essentiality_based_expression/Feature_collect/Feature_collect_result/Expression_across_cell_Line_normalized.rds")
weight_neig_exp=readRDS("/mnt/Storage/home/hanya/project/Predict_essentiality_based_expression/Feature_collect/Feature_collect_result/Gene_top10_neigh_exc_self_weight_exp.rds")
colnames(weight_neig_exp)[3:12]=paste0("Neig_",colnames(weight_neig_exp)[3:12])
Cell_line_pca=readRDS("/mnt/Storage/home/hanya/project/Predict_essentiality_based_expression/Feature_collect/Feature_collect_result/Exp_acr_normal_cell_line_sep_pca_feature.rds")
nest_score=readRDS("/mnt/Storage/home/hanya/project/Predict_essentiality_based_expression/Feature_collect/Feature_collect_result/Gene_itsel_NEST_expr_copy_mut.rds")
co_exp_ess=cbind(co_exp_ess,Cell_line_pca[,3:12],nest_score$Exp_Nest)
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
rm(Cell_line_pca)
rm(nest_score)

setwd("/mnt/Storage/home/hanya/project/Predict_essentiality_based_expression/Feature_selection_linear_model/Lasso_regression_for_each_gene")
#train_model_pre<-NULL
test_model_pre<-NULL
test_model_pre_cor<<-NULL
model<-list()
#If one columns only including one value, it will appeare error.
one_gene_model<-function(gene){
  print(gene)
  index=which(co_exp_ess[,2] == gene)
  data=co_exp_ess[index[1:385],3:35]
  essentiality=co_exp_ess[index[1:385],36]
  dtest=co_exp_ess[index[386:551],3:35]
  if(length(which(is.na(data)))>0){
    data=as.matrix(data)
    dtest=as.matrix(dtest)
    data[which(is.na(data))]=0
    dtest[which(is.na(dtest))]=0
    data=as.data.frame(data)
    dtest=as.data.frame(dtest)}
  temp_result=apply(as.matrix(data),2,function(x) table(x)[1])
  one_value_index=which(temp_result == 385)
  if(length(one_value_index) > 0){
    data=data[,-1*one_value_index]
    dtest=dtest[,-1*one_value_index]
  }
  data=as.matrix(data)
  dtest=as.matrix(dtest)
  lar1 <-lars(data,essentiality,type = "lasso")
  aa=summary(lar1)
  cp_index=aa$Df[which(aa$Cp == min(aa$Cp))]
  y_pre=predict(lar1,dtest,s=cp_index)
  test_model_pre<<-cbind(test_model_pre,y_pre$fit)
  test_model_pre_cor<<-rbind(test_model_pre_cor,c(gene,unlist(cor.test(y_pre$fit,co_exp_ess[index[386:551],36]))[c(3,4)]))
  model<<-c(model,gene<-list(lar1))
}
result=apply(matrix(gene_order),1,one_gene_model)

saveRDS(test_model_pre,file="Lar_Lasso_model_cor_net_nest_pca_test_predict.rds")
saveRDS(test_model_pre_cor,file="Lar_Lasso_model_cor_net_nest_pca_test_predict_gene_cor.rds")
saveRDS(model,file="Lar_Lasso_model_cor_net_nest_pca.rds")
