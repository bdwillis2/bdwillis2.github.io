# First let's go grab the libraries that are we likely to use here and rest of the env
options("width"=270)
library(dplyr)
library(melt)
library(caret)
library(data.table)
library(ggplot2)
library(e1071)
library(rpart)
library(randomForest)

# Now load the data that we want
hf_records_dt = fread("data/heart_failure_clinical_records_dataset.csv")

# Going to keep a semi-matrix based version of this, or at least one that is purely numbers and can be put into matrix format quickly
# for certain other tests
hf_records_raw_dt = hf_records_dt

correlation_data = round(cor(hf_records_dt), digits=2)
# Get upper triangle for viz
correlation_data[lower.tri(correlation_data)] = NA

hf_records_dt %>% head

# Let's start to see some of the possible data issues such as large numbers of NA's
hf_records_dt %>% summary

# So from what we can see there doesn't need to be any major NA issues, but there might be some better
# vector types for some of these values

# Going to convert some 0/1 values to TRUE FALSE for ease of looking and sampling inside the data
# Will look clearer in the summaries and the graphs that will be used to further explore the data

# First convert those that are best to be used as booleans
hf_records_dt$anaemia = as.logical(hf_records_dt$anaemia)
hf_records_dt$diabetes = as.logical(hf_records_dt$diabetes)
hf_records_dt$high_blood_pressure = as.logical(hf_records_dt$high_blood_pressure)
hf_records_dt$smoking = as.logical(hf_records_dt$smoking)
hf_records_dt$DEATH_EVENT = as.logical(hf_records_dt$DEATH_EVENT)

# Sex currently represented as a number, won't be consider a classification so
# we'll change it to a factor
hf_records_dt$sex = as.factor(as.character(hf_records_dt$sex))

# Finally lets change numeric to integer for age
hf_records_dt$age = as.integer(hf_records_dt$age)

# Quick check of the data
hf_records_dt %>% summary

# Heat map of the raw data
ggplot(data=reshape2::melt(correlation_data), aes(Var2, Var1, fill = value))+
 geom_tile(color = "white")+
 scale_fill_gradient2(low = "blue", high = "red", mid = "white",
   midpoint = 0, limit = c(-1,1), space = "Lab",
    name="Pearson\nCorrelation") +
  theme_minimal()+ # minimal theme
 theme(axis.text.x = element_text(angle = 45, vjust = 1,
    size = 12, hjust = 1))+
 coord_fixed() +
 geom_text(aes(Var2, Var1, label = value), color = "black", size = 4) +
theme(
  axis.title.x = element_blank(),
  axis.title.y = element_blank(),
  panel.grid.major = element_blank(),
  panel.border = element_blank(),
  panel.background = element_blank(),
  axis.ticks = element_blank(),
  legend.justification = c(1, 0),
  legend.position = c(0.6, 0.7),
  legend.direction = "horizontal")+
  guides(fill = guide_colorbar(barwidth = 7, barheight = 1,
                title.position = "top", title.hjust = 0.5))

hf_records_agg = setNames(
  aggregate(anaemia ~ age + DEATH_EVENT, hf_records_dt, length),
  c("age", "DEATH_EVENT", "count"))
ggplot(hf_records_agg, aes(x=age, y=count, fill=DEATH_EVENT)) +
  geom_bar(position="stack", stat="identity")

ggplot(hf_records_dt, aes(x=age,y=creatinine_phosphokinase, color=DEATH_EVENT))+
    geom_point()

ggplot(hf_records_dt, aes(x=age, y=ejection_fraction, color=DEATH_EVENT)) +
    geom_point()

ggplot(hf_records_dt, aes(x=age, y=platelets, color=DEATH_EVENT)) +
    geom_point()

ggplot(hf_records_dt, aes(x=age, y=serum_creatinine, color=DEATH_EVENT)) +
    geom_point()

ggplot(hf_records_dt, aes(x=age, y=serum_sodium, color=DEATH_EVENT)) +
    geom_point()

# Let's build some basic prediction models based on the data sets

# First let's identify some possible formulas here, going from complex with a bunch of stuff to simpler to prune
formula1 = DEATH_EVENT ~ age + anaemia + creatinine_phosphokinase + diabetes + ejection_fraction + high_blood_pressure + platelets + serum_creatinine + sex + serum_sodium + smoking
formula2 = DEATH_EVENT ~ age + anaemia + creatinine_phosphokinase + diabetes + ejection_fraction + high_blood_pressure + serum_creatinine + sex + serum_sodium + smoking

# Is it better to have the serum measure or just diabetes indicator?
formula3 = DEATH_EVENT ~ age + anaemia + creatinine_phosphokinase + diabetes + ejection_fraction + high_blood_pressure  + sex + serum_sodium + smoking
formula4 = DEATH_EVENT ~ age + anaemia + creatinine_phosphokinase  + ejection_fraction + high_blood_pressure + serum_creatinine + sex + serum_sodium + smoking

# Logisitic regression, multiple kinds of distributions since this isn't exactly a "normal" distrobution
runLogisticModeling = function(data, formula, split_p=0.8, ...){
  # Going to split off and create 4 indicex splits
  index1 = sample(1:nrow(data), size=round(nrow(data)*split_p))
  index2 = sample(1:nrow(data), size=round(nrow(data)*split_p))
  index3 = sample(1:nrow(data), size=round(nrow(data)*split_p))
  index4 = sample(1:nrow(data), size=round(nrow(data)*split_p))
  index5 = sample(1:nrow(data), size=round(nrow(data)*split_p))
  index6 = sample(1:nrow(data), size=round(nrow(data)*split_p))
  index7 = sample(1:nrow(data), size=round(nrow(data)*split_p))
  index8 = sample(1:nrow(data), size=round(nrow(data)*split_p))
  index9 = sample(1:nrow(data), size=round(nrow(data)*split_p))
  index10 = sample(1:nrow(data), size=round(nrow(data)*split_p))

  # For each one of these we are going to bebuilding the glm logisitic model
  model1 = glm(formula, data[index1,], family=binomial(link="logit"), ...)
  model2 = glm(formula, data[index2,], family=binomial(link="logit"), ...)
  model3 = glm(formula, data[index3,], family=binomial(link="logit"), ...)
  model4 = glm(formula, data[index4,], family=binomial(link="logit"), ...)
  model5 = glm(formula, data[index5,], family=binomial(link="logit"), ...)
  model6 = glm(formula, data[index6,], family=binomial(link="logit"), ...)
  model7 = glm(formula, data[index7,], family=binomial(link="logit"), ...)
  model8 = glm(formula, data[index8,], family=binomial(link="logit"), ...)
  model9 = glm(formula, data[index9,], family=binomial(link="logit"), ...)
  model10 = glm(formula, data[index10,], family=binomial(link="logit"), ...)

  # Now we are going to run the various predictions
  predict1 = round(predict(model1, data[!index1,], type="response"))
  predict2 = round(predict(model2, data[!index2,], type="response"))
  predict3 = round(predict(model3, data[!index3,], type="response"))
  predict4 = round(predict(model4, data[!index4,], type="response"))
  predict5 = round(predict(model5, data[!index5,], type="response"))
  predict6 = round(predict(model6, data[!index6,], type="response"))
  predict7 = round(predict(model7, data[!index7,], type="response"))
  predict8 = round(predict(model8, data[!index8,], type="response"))
  predict9 = round(predict(model9, data[!index9,], type="response"))
  predict10 = round(predict(model10, data[!index10,], type="response"))

  # Now for confusion matrices
  conf1 = confusionMatrix(table(predict1, as.integer(unlist(data[!index1, "DEATH_EVENT"]))))
  conf2 = confusionMatrix(table(predict2, as.integer(unlist(data[!index2, "DEATH_EVENT"]))))
  conf3 = confusionMatrix(table(predict3, as.integer(unlist(data[!index3, "DEATH_EVENT"]))))
  conf4 = confusionMatrix(table(predict4, as.integer(unlist(data[!index4, "DEATH_EVENT"]))))
  conf5 = confusionMatrix(table(predict5, as.integer(unlist(data[!index5, "DEATH_EVENT"]))))
  conf6 = confusionMatrix(table(predict6, as.integer(unlist(data[!index6, "DEATH_EVENT"]))))
  conf7 = confusionMatrix(table(predict7, as.integer(unlist(data[!index7, "DEATH_EVENT"]))))
  conf8 = confusionMatrix(table(predict8, as.integer(unlist(data[!index8, "DEATH_EVENT"]))))
  conf9 = confusionMatrix(table(predict9, as.integer(unlist(data[!index9, "DEATH_EVENT"]))))
  conf10 = confusionMatrix(table(predict10, as.integer(unlist(data[!index10, "DEATH_EVENT"]))))

  return(mean(conf1$overall[1],conf2$overall[1],conf3$overall[1],conf4$overall[1],conf5$overall[1],conf6$overall[1],conf7$overall[1],conf8$overall[1],conf9$overall[1],conf10$overall[1]))
}

getAvgLogisticAccuracy = function(data, formula, p_split=0.8, iters=1000){
  accs = sapply(1:iters, function(x, p_df, p_split){
    y = x
    index = sample(1:nrow(p_df), size=round(nrow(p_df)*p_split))
    model = glm(formula, p_df[index,], family=binomial(link="logit"))
    predict = round(predict(model, p_df[!index,], type="response"))
    conf = confusionMatrix(table(predict, as.integer(unlist(p_df[!index, "DEATH_EVENT"]))))
    conf$overall[1]
  }, p_df=data, p_split=p_split)
  print(paste("Average accuracy over", iters, "iterations:",mean(accs)))
}

runLogisticModeling(hf_records_dt, formula1)
runLogisticModeling(hf_records_dt, formula2)
runLogisticModeling(hf_records_dt, formula3)
runLogisticModeling(hf_records_dt, formula4)

# Random forest algorithm?
# Not able to since there are so few types of outcomes, instead we just have yes or no

getAvgDecTreeAccuracy = function(data, formula, p_split=0.8, iters=1000){
  accs = sapply(1:iters, function(x, p_df, p_split){
    y = x
    index = sample(1:nrow(p_df), size=round(nrow(p_df)*p_split))
    model = rpart(formula, p_df[index,])
    predict = round(predict(model, p_df[!index,]))
    conf = confusionMatrix(table(predict, as.integer(unlist(p_df[!index, "DEATH_EVENT"]))))
    conf$overall[1]
  }, p_df=data, p_split=p_split)
  print(paste("Average accuracy over", iters, "iterations:",mean(accs)))
}

# Now hit is up with basic decision tree, this just can take the entire dataset
runDecisionTreeModeling = function(data, formula, split_p=0.8, ...){
  # Going to split off and create 4 indicex splits
  index1 = sample(1:nrow(data), size=round(nrow(data)*split_p))
  index2 = sample(1:nrow(data), size=round(nrow(data)*split_p))
  index3 = sample(1:nrow(data), size=round(nrow(data)*split_p))
  index4 = sample(1:nrow(data), size=round(nrow(data)*split_p))
  index5 = sample(1:nrow(data), size=round(nrow(data)*split_p))
  index6 = sample(1:nrow(data), size=round(nrow(data)*split_p))
  index7 = sample(1:nrow(data), size=round(nrow(data)*split_p))
  index8 = sample(1:nrow(data), size=round(nrow(data)*split_p))
  index9 = sample(1:nrow(data), size=round(nrow(data)*split_p))
  index10 = sample(1:nrow(data), size=round(nrow(data)*split_p))

  # For each one of these we are going to bebuilding the glm logisitic model
  model1 = rpart(formula, data[index1,], ...)
  model2 = rpart(formula, data[index2,], ...)
  model3 = rpart(formula, data[index3,], ...)
  model4 = rpart(formula, data[index4,], ...)
  model5 = rpart(formula, data[index5,], ...)
  model6 = rpart(formula, data[index6,], ...)
  model7 = rpart(formula, data[index7,], ...)
  model8 = rpart(formula, data[index8,], ...)
  model9 = rpart(formula, data[index9,], ...)
  model10 = rpart(formula, data[index10,], ...)

  # Now we are going to run the various predictions
  predict1 = round(predict(model1, data[!index1,]))
  predict2 = round(predict(model2, data[!index2,]))
  predict3 = round(predict(model3, data[!index3,]))
  predict4 = round(predict(model4, data[!index4,]))
  predict5 = round(predict(model5, data[!index5,]))
  predict6 = round(predict(model6, data[!index6,]))
  predict7 = round(predict(model7, data[!index7,]))
  predict8 = round(predict(model8, data[!index8,]))
  predict9 = round(predict(model9, data[!index9,]))
  predict10 = round(predict(model10, data[!index10,]))

  # Now for confusion matrices
  conf1 = confusionMatrix(table(predict1, as.integer(unlist(data[!index1, "DEATH_EVENT"]))))
  conf2 = confusionMatrix(table(predict2, as.integer(unlist(data[!index2, "DEATH_EVENT"]))))
  conf3 = confusionMatrix(table(predict3, as.integer(unlist(data[!index3, "DEATH_EVENT"]))))
  conf4 = confusionMatrix(table(predict4, as.integer(unlist(data[!index4, "DEATH_EVENT"]))))
  conf5 = confusionMatrix(table(predict5, as.integer(unlist(data[!index5, "DEATH_EVENT"]))))
  conf6 = confusionMatrix(table(predict6, as.integer(unlist(data[!index6, "DEATH_EVENT"]))))
  conf7 = confusionMatrix(table(predict7, as.integer(unlist(data[!index7, "DEATH_EVENT"]))))
  conf8 = confusionMatrix(table(predict8, as.integer(unlist(data[!index8, "DEATH_EVENT"]))))
  conf9 = confusionMatrix(table(predict9, as.integer(unlist(data[!index9, "DEATH_EVENT"]))))
  conf10 = confusionMatrix(table(predict10, as.integer(unlist(data[!index10, "DEATH_EVENT"]))))

  return(mean(conf1$overall[1],conf2$overall[1],conf3$overall[1],conf4$overall[1],conf5$overall[1],conf6$overall[1],conf7$overall[1],conf8$overall[1],conf9$overall[1],conf10$overall[1]))
}

runDecisionTreeModeling(hf_records_dt, formula1)
runDecisionTreeModeling(hf_records_dt, formula2)
runDecisionTreeModeling(hf_records_dt, formula3)
runDecisionTreeModeling(hf_records_dt, formula4)


getAvgSVMAccuracy = function(data, formula, p_split=0.8, iters=1000, ...){
  accs = sapply(1:iters, function(x, p_df, p_split){
    y = x
    index = sample(1:nrow(p_df), size=round(nrow(p_df)*p_split))
    model = svm(formula, data[index,], type="one-classification", ...)
    predict = predict(model, data[!index,])
    conf = confusionMatrix(table(predict, as.logical(unlist(p_df[!index, "DEATH_EVENT"]))))
    conf$overall[1]
  }, p_df=data, p_split=p_split)
  print(paste("Average accuracy over", iters, "iterations:",mean(accs)))
}

# SVM? It is possible here
runSVMModeling = function(data, formula, split_p=0.8, ...){
  # Going to split off and create 4 indicex splits
  index1 = sample(1:nrow(data), size=round(nrow(data)*split_p))
  index2 = sample(1:nrow(data), size=round(nrow(data)*split_p))
  index3 = sample(1:nrow(data), size=round(nrow(data)*split_p))
  index4 = sample(1:nrow(data), size=round(nrow(data)*split_p))
  index5 = sample(1:nrow(data), size=round(nrow(data)*split_p))
  index6 = sample(1:nrow(data), size=round(nrow(data)*split_p))
  index7 = sample(1:nrow(data), size=round(nrow(data)*split_p))
  index8 = sample(1:nrow(data), size=round(nrow(data)*split_p))
  index9 = sample(1:nrow(data), size=round(nrow(data)*split_p))
  index10 = sample(1:nrow(data), size=round(nrow(data)*split_p))

  # For each one of these we are going to bebuilding the glm logisitic model
  model1  = svm(formula, data[index1,], type="one-classification", ...)
  model2  = svm(formula, data[index2,], type="one-classification", ...)
  model3  = svm(formula, data[index3,], type="one-classification", ...)
  model4  = svm(formula, data[index4,], type="one-classification", ...)
  model5  = svm(formula, data[index5,], type="one-classification", ...)
  model6  = svm(formula, data[index6,], type="one-classification", ...)
  model7  = svm(formula, data[index7,], type="one-classification", ...)
  model8  = svm(formula, data[index8,], type="one-classification", ...)
  model9  = svm(formula, data[index9,], type="one-classification", ...)
  model10 = svm(formula, data[index10,], type="one-classification", ...)

  # Now we are going to run the various predictions
  predict1 = predict(model1, data[!index1,])
  predict2 = predict(model2, data[!index2,])
  predict3 = predict(model3, data[!index3,])
  predict4 = predict(model4, data[!index4,])
  predict5 = predict(model5, data[!index5,])
  predict6 = predict(model6, data[!index6,])
  predict7 = predict(model7, data[!index7,])
  predict8 = predict(model8, data[!index8,])
  predict9 = predict(model9, data[!index9,])
  predict10 = predict(model10, data[!index10,])

  # Now for confusion matrices
  conf1 = confusionMatrix(table(predict1, as.logical(unlist(data[!index1, "DEATH_EVENT"]))))
  conf2 = confusionMatrix(table(predict2, as.logical(unlist(data[!index2, "DEATH_EVENT"]))))
  conf3 = confusionMatrix(table(predict3, as.logical(unlist(data[!index3, "DEATH_EVENT"]))))
  conf4 = confusionMatrix(table(predict4, as.logical(unlist(data[!index4, "DEATH_EVENT"]))))
  conf5 = confusionMatrix(table(predict5, as.logical(unlist(data[!index5, "DEATH_EVENT"]))))
  conf6 = confusionMatrix(table(predict6, as.logical(unlist(data[!index6, "DEATH_EVENT"]))))
  conf7 = confusionMatrix(table(predict7, as.logical(unlist(data[!index7, "DEATH_EVENT"]))))
  conf8 = confusionMatrix(table(predict8, as.logical(unlist(data[!index8, "DEATH_EVENT"]))))
  conf9 = confusionMatrix(table(predict9, as.logical(unlist(data[!index9, "DEATH_EVENT"]))))
  conf10 = confusionMatrix(table(predict10, as.logical(unlist(data[!index10, "DEATH_EVENT"]))))

  return(mean(conf1$overall[1],conf2$overall[1],conf3$overall[1],conf4$overall[1],conf5$overall[1],conf6$overall[1],conf7$overall[1],conf8$overall[1],conf9$overall[1],conf10$overall[1]))
}

runSVMModeling(hf_records_raw_dt, formula1)
runSVMModeling(hf_records_raw_dt, formula2)
runSVMModeling(hf_records_raw_dt, formula3)
runSVMModeling(hf_records_raw_dt, formula4)

runSVMModeling(hf_records_raw_dt, formula1, kernel="linear")
runSVMModeling(hf_records_raw_dt, formula2, kernel="linear")
runSVMModeling(hf_records_raw_dt, formula3, kernel="linear")
runSVMModeling(hf_records_raw_dt, formula4, kernel="linear")

runSVMModeling(hf_records_raw_dt, formula1, kernel="polynomial")
runSVMModeling(hf_records_raw_dt, formula2, kernel="polynomial")
runSVMModeling(hf_records_raw_dt, formula3, kernel="polynomial")
runSVMModeling(hf_records_raw_dt, formula4, kernel="polynomial")

runSVMModeling(hf_records_raw_dt, formula1, kernel="sigmoid")
runSVMModeling(hf_records_raw_dt, formula2, kernel="sigmoid")
runSVMModeling(hf_records_raw_dt, formula3, kernel="sigmoid")
runSVMModeling(hf_records_raw_dt, formula4, kernel="sigmoid")

# Getting data and results for many many iterations for each model

getAvgLogisticAccuracy(hf_records_dt, formula1, iters=10000)
getAvgLogisticAccuracy(hf_records_dt, formula2, iters=10000)
getAvgLogisticAccuracy(hf_records_dt, formula3, iters=10000)
getAvgLogisticAccuracy(hf_records_dt, formula4, iters=10000)

getAvgDecTreeAccuracy(hf_records_dt, formula1, iters=10000)
getAvgDecTreeAccuracy(hf_records_dt, formula2, iters=10000)
getAvgDecTreeAccuracy(hf_records_dt, formula3, iters=10000)
getAvgDecTreeAccuracy(hf_records_dt, formula4, iters=10000)

getAvgSVMAccuracy(hf_records_raw_dt, formula1, iters=10000)
getAvgSVMAccuracy(hf_records_raw_dt, formula2, iters=10000)
getAvgSVMAccuracy(hf_records_raw_dt, formula3, iters=10000)
getAvgSVMAccuracy(hf_records_raw_dt, formula4, iters=10000)

getAvgSVMAccuracy(hf_records_raw_dt, formula1, kernel="linear", iters=10000)
getAvgSVMAccuracy(hf_records_raw_dt, formula2, kernel="linear", iters=10000)
getAvgSVMAccuracy(hf_records_raw_dt, formula3, kernel="linear", iters=10000)
getAvgSVMAccuracy(hf_records_raw_dt, formula4, kernel="linear", iters=10000)

getAvgSVMAccuracy(hf_records_raw_dt, formula1, kernel="polynomial", iters=10000)
getAvgSVMAccuracy(hf_records_raw_dt, formula2, kernel="polynomial", iters=10000)
getAvgSVMAccuracy(hf_records_raw_dt, formula3, kernel="polynomial", iters=10000)
getAvgSVMAccuracy(hf_records_raw_dt, formula4, kernel="polynomial", iters=10000)

getAvgSVMAccuracy(hf_records_raw_dt, formula1, kernel="sigmoid", iters=10000)
getAvgSVMAccuracy(hf_records_raw_dt, formula2, kernel="sigmoid", iters=10000)
getAvgSVMAccuracy(hf_records_raw_dt, formula3, kernel="sigmoid", iters=10000)
getAvgSVMAccuracy(hf_records_raw_dt, formula4, kernel="sigmoid", iters=10000)
