#### Carico il dataset #### 
Data=read.csv("RegressionDSDA250130.csv",header=T)
head(Data)
dim(Data) #1 è la variabile dipendente

# valutiamo se esistono variabbili non segnate (non ci sono)
dim(na.omit(Data))
Data=na.omit(Data)
dim(Data)

n = dim(Data)[1] #n di samples
names(Data)
attach(Data)

# Creazione del test set
set.seed(2021)
x = model.matrix(Y~., Data)[,-1] # without 1's, senza la y
y = Data$Y
train=sample(1:nrow(x), 0.7*nrow(x)) # 70% training set
test=(-train)
y.test=y[test]

d_train = Data[train,] #70% train e 20% test
d_test = Data[test,]




# VALUTAZIONE CORRELAZIONE LINEARE
dev.new()
pairs(Data) #sono troppi regressori

corBo <- round(cor(Data), digits =2) #MATRICE DI CORRELAZIONE CON co

library(corrplot)
dev.new()
corrplot(corBo,method = 'ellipse')
dev.print(device=pdf, "corplot.pdf")
dev.off()

dev.new()
corrplot.mixed(corBo,order='original',number.cex=1, upper="ellipse")
print()


# Valutazione multi-colinearità
fit = lm(Y~., data=d_train)
library(car)
vif(fit)
dev.new(); par(mfrow=c(2,2)) 
plot(fit)
dev.print(device=pdf, "fit.pdf")

# vediamo che possiamo considerare gli errori iid grazie al grafico 1, possiamo suppore gaussianità col grafico 2, no outlier o punti di leva 



#BSS
library(ISLR); library(leaps)
regfit.bss=regsubsets(Y~.,data=d_train, nvmax=25) # massimo 25 regressori
bss_summ = summary(regfit.bss)
dev.new()
plot(bss_summ$bic ,xlab="Number of Variables ",ylab="BIC",type="l") # facciamo il grafico
points(which.min(bss_summ$bic),min(bss_summ$bic),col="red",cex=2,pch=20) # non valutiamo la one standard rule essendo già il modello a 4 regressori
dev.print(device=pdf, "bss.pdf")
min=which.min(bss_summ$bic); min #mi dice il grado del minore BIC
print(coef(regfit.bss,which.min(bss_summ$bic)))
print(coef(regfit.bss,min)) #per one standard rule potrei prender anche quello a 15






### CROSS VALIDATION Function###
predict.regsubsets = function(object,newdata,id,...){ # ... <-> ellipsis significa che se voglio posso passare altri parametri
  form=as.formula(object$call[[2]])
  mat=model.matrix(form, newdata) #a partire dalla formula mi calcolo la model matrix perche qua non so quanti coefficienti ho
  coefi=coef(object, id=id)
  xvars=names(coefi)
  mat[,xvars]%*%coefi #prendo solo i coefficienti diversi da 0
}

#CROSS-VALIDATION CON backward
k=5
set.seed(2021)
folds = sample(1:k,nrow(d_train),replace=FALSE)
#folds = sample(rep(1:k, length.out = nrow(d_train)))
cv.errors = matrix(NA,k,25, dimnames=list(NULL, paste(1:25)))
for(j in 1:k){ #da 1 a k
  best.fit=regsubsets(Y~., data=d_train[folds!=j,], nvmax=25,method="backward") #training usa tutti i fold tranne il j-esimo
  for(i in 1:25){#calcola l'errore per ogni modello
    pred = predict(best.fit, d_train[folds==j,], id=i) #poi per il j-esimo fa il test
    cv.errors[j,i] = mean((d_train$Y[folds==j]-pred)^2)
    
  }
}

mean.cv.errors=apply(cv.errors, 2, mean); mean.cv.errors

dev.new()
plot(mean.cv.errors, type="b") #il minimo è 4
min_index <- which.min(mean.cv.errors)  # Trova l'indice del minimo
min_value <- mean.cv.errors[min_index]  # Trova il valore del minimo
points(min_index, min_value, col="red", pch=19)
dev.print(device=pdf, "bacward.pdf")
reg.best=regsubsets (Y~., data=d_train, nvmax=25)
coef(reg.best, which.min(mean.cv.errors)) #COEFFICIENTI BSS

library(glmnet)
library(ISLR)
library(plotmo)

#Ridge cross-validation
cv.out=cv.glmnet(x[train,],y[train],alpha=0) #posso fare lambda=grid oppure lasciare il valore di default, nella documentazione
dev.new();
plot(cv.out)
bestlam=cv.out$lambda.min; bestlam; log(bestlam) #miglior lambda
cv.out$lambda.1se; log(cv.out$lambda.1se)#miglior lambda per la standard rule
dev.print(device=pdf, "ridge.pdf")

#Lasso cross-validation
lasso.out=cv.glmnet(x[train,],y[train],alpha=1)
dev.new();
plot(lasso.out) #valori di lambda
bestlam2=lasso.out$lambda.min; bestlam2; log(bestlam2) #miglior lambda
lasso.out$lambda.1se#miglior lambda per la standard rule
dev.print(device=pdf, "lasso.pdf")

#TROVO GLI MSE SUL TEST###
#ridge
ridge.pred=predict(cv.out,s=bestlam ,newx=x[test,])
mse_ridge <- mean((ridge.pred-y.test)^2); mse_ridge
coef(cv.out,s=bestlam)


#lasso
lasso.pred=predict(lasso.out,s=bestlam2 ,newx=x[test,])
mse_lasso <- mean((lasso.pred-y.test)^2); mse_lasso
coef(lasso.out,s=bestlam2)

lasso.pred=predict(lasso.out,s=lasso.out$lambda.1se ,newx=x[test,])
mse_lasso <- mean((lasso.pred-y.test)^2); mse_lasso
coef(lasso.out,s=lasso.out$lambda.1se)

#bss, ATTENZIONE QUESTO NON E' CON CROSS VALIDATION!
bss_pred= predict(regfit.bss, Data[test,], id=which.min(bss_summ$bic))
mse_bss = mean((Data$Y[test]-bss_pred)^2)
mse_bss

#backward
bwd_pred= predict(reg.best, Data[test,], id=which.min(mean.cv.errors))
mse_bwd = mean((Data$Y[test]-bwd_pred)^2)
mse_bwd


# Our model

fit = lm(Y~X2+X4+X18+X25, data=d_train)
library(car)
vif(fit)
dev.new(); par(mfrow=c(2,2)) 
plot(fit)
dev.print(device=pdf, "fit.pdf")


modelx=lm(Y~X2+X4+X9+X18+X25, data=Data)
summary(modelx)

modelx= predict(modelx, Data[test,])
mse_modelx = mean((Data$Y[test]-modelx)^2)
mse_modelx

