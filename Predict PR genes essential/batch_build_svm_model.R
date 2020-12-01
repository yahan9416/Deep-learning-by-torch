library(e1071)
correaltion_coeff<-NULL
feature_matrix=readRDS("/mnt/Storage/home/hanya/project/Predict_essentiality_based_expression/PR_Gene_ten_feature_SVM/SVM_Model_for_each_gene_Dream_data/Train_SVM_model_feature_matrix.rds")
test_dataset=readRDS("/mnt/Storage/home/hanya/project/Predict_essentiality_based_expression/PR_Gene_ten_feature_SVM/SVM_Model_for_each_gene_Dream_data/Test_SVM_model_feature_matrix.rds")
generate_SVM_model_for_each_gene<-function(x){
  index=which(feature_matrix[,1] == x)
  temp_feature_mat=feature_matrix[index,3:13]
  col_tem=colnames(temp_feature_mat)
  temp_feature_mat=matrix(as.numeric(as.vector(unlist(temp_feature_mat))),ncol=11,byrow=F)
  temp_feature_mat=data.frame(temp_feature_mat)
  colnames(temp_feature_mat)=col_tem
  
  index_test=which(test_dataset[,1] == x)
  temp_test_fea=test_dataset[index_test,3:12]
  temp_test_fea=data.frame(matrix(as.numeric(as.vector(unlist(temp_test_fea))),ncol=10,byrow = F))
  colnames(temp_test_fea)=col_tem[-11]
  temp_test_essential=as.numeric(as.vector(unlist(test_dataset[index_test,13])))
  
  #for each gene is do 5 round 5-cross validation. so for each gene we need to build 25
  round_5_cross_5<-function(y){
    set.seed(y)
    index=sample(1:105)
    temp_feature_mat=temp_feature_mat[index,]
    model_one_gene_1=svm(Essentiality~.,data=temp_feature_mat[-1*(1:21),],type="eps-regression")
    pred_essential<-predict(model_one_gene_1,temp_test_fea)
    cor_coeff<-cor.test(pred_essential,temp_test_essential,method="spearman")
    correaltion_coeff<<-rbind(correaltion_coeff,c(x,cor_coeff$estimate,cor_coeff$p.value))
    model_one_gene_2=svm(Essentiality~.,data=temp_feature_mat[-1*(22:42),],type="eps-regression")
    pred_essential<-predict(model_one_gene_2,temp_test_fea)
    cor_coeff<-cor.test(pred_essential,temp_test_essential,method="spearman")
    correaltion_coeff<<-rbind(correaltion_coeff,c(x,cor_coeff$estimate,cor_coeff$p.value))
    model_one_gene_3=svm(Essentiality~.,data=temp_feature_mat[-1*(43:63),],type="eps-regression")
    pred_essential<-predict(model_one_gene_3,temp_test_fea)
    cor_coeff<-cor.test(pred_essential,temp_test_essential,method="spearman")
    correaltion_coeff<<-rbind(correaltion_coeff,c(x,cor_coeff$estimate,cor_coeff$p.value))
    model_one_gene_4=svm(Essentiality~.,data=temp_feature_mat[-1*(64:84),],type="eps-regression")
    pred_essential<-predict(model_one_gene_4,temp_test_fea)
    cor_coeff<-cor.test(pred_essential,temp_test_essential,method="spearman")
    correaltion_coeff<<-rbind(correaltion_coeff,c(x,cor_coeff$estimate,cor_coeff$p.value))
    model_one_gene_5=svm(Essentiality~.,data=temp_feature_mat[-1*(85:105),],type="eps-regression")
    pred_essential<-predict(model_one_gene_5,temp_test_fea)
    cor_coeff<-cor.test(pred_essential,temp_test_essential,method="spearman")
    correaltion_coeff<<-rbind(correaltion_coeff,c(x,cor_coeff$estimate,cor_coeff$p.value))
    saveRDS(model_one_gene_1,paste0("/mnt/Storage/home/hanya/project/Predict_essentiality_based_expression/PR_Gene_ten_feature_SVM/SVM_Model_for_each_gene_Dream_data/SVM_model_for_each_gene/","RPgene_",x[1],"_Round_",y,"model_1.rds"))
    saveRDS(model_one_gene_2,paste0("/mnt/Storage/home/hanya/project/Predict_essentiality_based_expression/PR_Gene_ten_feature_SVM/SVM_Model_for_each_gene_Dream_data/SVM_model_for_each_gene/","RPgene_",x[1],"_Round_",y,"model_2.rds"))
    saveRDS(model_one_gene_3,paste0("/mnt/Storage/home/hanya/project/Predict_essentiality_based_expression/PR_Gene_ten_feature_SVM/SVM_Model_for_each_gene_Dream_data/SVM_model_for_each_gene/","RPgene_",x[1],"_Round_",y,"model_3.rds"))
    saveRDS(model_one_gene_4,paste0("/mnt/Storage/home/hanya/project/Predict_essentiality_based_expression/PR_Gene_ten_feature_SVM/SVM_Model_for_each_gene_Dream_data/SVM_model_for_each_gene/","RPgene_",x[1],"_Round_",y,"model_4.rds"))
    saveRDS(model_one_gene_5,paste0("/mnt/Storage/home/hanya/project/Predict_essentiality_based_expression/PR_Gene_ten_feature_SVM/SVM_Model_for_each_gene_Dream_data/SVM_model_for_each_gene/","RPgene_",x[1],"_Round_",y,"model_5.rds"))
    
  }
  result=apply(matrix(1:5),1,round_5_cross_5)
  
}
pr_genes=matrix(unique(as.vector(unlist(feature_matrix[,1]))))
result=apply(matrix(pr_genes[1:1300]),1,generate_SVM_model_for_each_gene)

saveRDS(correaltion_coeff,file="/mnt/Storage/home/hanya/project/Predict_essentiality_based_expression/PR_Gene_ten_feature_SVM/SVM_Model_for_each_gene_Dream_data/SVM_model_for_each_gene/PRgenes_SVM_model_test_dataset_correlation.rds")
