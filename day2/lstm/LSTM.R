##----------------------------------------------------------
##
##  Forecasting Life expectancy: LSTM vs ARIMA vs Double gap
##  ITALY Female - Periodo di applicazione [1921:2016]
##  
##-----------------------------------------------------------

# Library
library(demography)
library(HMDHFDplus)
library(tidyverse)
library(scales)
library(keras)
library(tensorflow)
library(forecast)
library(ggplot2)
source("FunctionHMD.R")

# Error functions 

rmse = function (truth, prediction)  {
  sqrt(mean((prediction - truth)^2))
}
mae = function(truth, prediction){
  mean(abs(prediction-truth))
}


###--------------------------------------------------------------------------###
#                                                                             ##
# MODELLO DOUBLE GAP (DG) - Pasacrius et al. (2017)                           ##
#                                                                             ##
###--------------------------------------------------------------------------###

# Caricamento dei dati coerentemente con la strutturazione del package del DG

devtools::install_github("mpascariu/MortalityGaps")
devtools::install_github("mpascariu/MortalityLaws")
install.packages("MortalityLaws")
install.packages("MortalityGaps")
library(MortalityLaws)
library(MortalityGaps)
library(dplyr)

#user_ = 'andrea.nigri@student.unisi.it'
#password_ = 'Dottorato17'

# Download HMD data. Take approx 1m30s (depending on the internet speed).
#HMD_LT_F <- ReadHMD(what = 'LT_f',
#                    interval = '1x1',
#                   username = user_,
#                   password = password_,
#                   save = TRUE)

#HMD_LT_M <- ReadHMD(what = 'LT_m',
#                   interval = '1x1',
#                   username = user_,
#                   password = password_,
#                   save = TRUE)

# cnu = countries not used  
# these are populations that need to be taken out of the dataset
# Da utilizzare solo se si vuole fissare il periodo storico dell'analisi
#LTF <- HMD_LT_F$data %>% filter(Year %in% years & !(country %in% cnu))
#LTM <- HMD_LT_M$data %>% filter(Year %in% years & !(country %in% cnu))
##############
#DATI CARICATI ON LINE
#LTF <- HMD_LT_F$data %>% filter( !(country %in% cnu))
#LTM <- HMD_LT_M$data %>% filter( !(country %in% cnu))

load(file = "HMD_LT_f.Rdata")
load(file = "HMD_LT_m.Rdata")
cnu <- c("FRACNP","DEUTNP", "NZL_NM", "NZL_MA",
         "GBR_NP","GBRCENW", "GBR_NIR", "CHL", "LUX", "HRV")

LTF <- HMD_LT_f$data %>% filter( !(country %in% cnu))
LTM <- HMD_LT_m$data %>% filter( !(country %in% cnu))


exF <- LTF %>% filter(Age %in% c(0, 65)) %>% select(country, Year, Age, ex)
exM <- LTM %>% filter(Age %in% c(0, 65)) %>% select(country, Year, Age, ex)
# verify that the two data.frames are of equal length
nrow(exF) == nrow(exM)
ncol(exF) == ncol(exM)

MortalityGaps.data <- structure(class = "MortalityGaps.data", list(exF = exF, exM = exM))

head(exF)
head(exM)

###-------------------------------------------------###
#             Analisi life exp a 65                   #
###-------------------------------------------------###

exM_65<- LTM %>% filter(Age ==65)

# Creo l'oggetto contenente le life exp a 65 anni per ogni anni e solo per l'Australia 
lf_M <-exM_65%>% filter(country=="ITA")

# Selezione della life exp a 65 anni sfruttando le funzioni di lettura del pacchetto di Pascarius
# in particolare questo comando carica il valore della life table a 65 anni
# per ogni anno di calendario e per ogni paese
# vengono considerate qu' le DONNE
exF_65<- LTF %>% filter(Age ==65)

# Creo l'oggetto contenente le life exp a 65 anni per ogni anni e solo per l'Australia 
lf_F <-exF_65%>% filter(country=="ITA")
head(lf_F)
y <- lf_F$Year
#scelta del sesso: DONNE. Rinomino l'oggetto life exp "exp_f" 
exp_f<- lf_F$ex
lifexp = as.numeric(exp_f)
plot(y,exp_f)


#---------------------------------------------------------------------------------------#
#                         FITTING LSTM + FITTING AND FORECAST ARIMA                     #
#---------------------------------------------------------------------------------------#

#------ Creazione train e test sugli anni con criterio split 80%-20% ---------------
start = y[1]
finish = y[length(y)]
brek = finish- round(((finish- start)*0.2))
year <- seq(start,finish,1)

#----- Creazione del Dataset SUPERVISED (con 1 solo Temporal Lag) ------------
n =(brek - start)+1
L = finish -brek
numberlag = 1
supervised = lifexp
for(i in 1:numberlag) {
  lag = c(rep(lifexp[1],i),lifexp[1:(length(lifexp)-i)])
  supervised = cbind(supervised,lag)}
train = supervised[1:n, ]
test = supervised[(n+1):nrow(supervised),]
x_train = train[,-1]
y_train = train[,1]
x_test = test[,-1]
y_test = test[,1]


#--------- Fit and forecast BEST ARIMA  ----------------
fit = auto.arima(y_train, stepwise = F, approximation = F)
fit
predict_arima = forecast(fit, h = L)


#--------- TRAINING della LSTM -----------------------
dim(x_train) <- c(n, (numberlag), 1)
dim(x_train)
X_shape2 = numberlag
X_shape3 = 1
batch_size = 1
units = 1

# utile per gestire la calibrazione della rete presupponendo un ampio (e automatizzato) spazio degli iper-parametri
# results = expand.grid(epochs = seq(5,50,5), unit = c(seq(50, 200, 10)))
# results = cbind(results, performance = rep(0, nrow(results)))


# Campionamento del seed
set.seed(50)
seed= round(runif(10, min = 0, max = 1000 ))
seed

# Creazione matrice degli errori 
tuttirmse_arima = numeric(length(seed))
tuttirmse_lstm = numeric(length(seed))
matprediction = matrix(rep(0,length(seed)*L), nrow = length(seed), ncol = L)

# Ciclo per ripetere fitting (e successivo forecast) su ogni seed campionato
# NOTA: Fine tuning "manuale" su # neuroni e epochs.
# Fissato 1 solo layer con activation function = relu
set.seed(2000)
for( k in 1:length(seed)){
  
  library(keras)
  use_session_with_seed(seed[k])
  build_model <- function() {
    
    model <- keras_model_sequential() 
    model%>%
      layer_lstm(30,batch_input_shape = c(batch_size, X_shape2, X_shape3),activation='relu', recurrent_activation = "tanh")%>%
      layer_dense(units = 1)
    
    model %>% compile(
      loss = 'mean_squared_error',
      optimizer = optimizer_adadelta(),  
      metrics = c('accuracy')
      
    )
    model
  }
  
  model <- build_model()
  model %>% summary()
  h <- model %>% fit(x_train, 
                     y_train,
                     epochs=4,
                     batch_size=1, 
                     verbose=0, 
                     shuffle=FALSE)
  dim(x_train)
  dim(x_test)<-c((finish-brek),numberlag,1)
  
  predict_train= model %>% predict(x_train, batch_size=batch_size)
  predict_train
  rmse(predict_train, y_train)
  
  
  #---------------------------------------------------------------------------------------#
  #                              FORECAST (SENZA RE-TRAIN)                                #
  #---------------------------------------------------------------------------------------#
  
  L = finish-brek
  predict_test = numeric(L)
  X = c(tail(y_train, n = 1), x_train[dim(x_train)[1],1:(ncol(x_train)-1),1])
  X = tail(y_train, n = 1)
  for(i in 1:L){
    X
    dim(X) = c(1,numberlag,1)
    yhat = model %>% predict(X, batch_size=batch_size)
    yhat
    predict_test[i] = yhat
    #X = c(yhat, X[-3])
    X = yhat
  }
  
  matprediction[k,] = predict_test
  tuttirmse_lstm[k] = rmse(predict_test, y_test)
  tuttirmse_arima[k] = rmse(predict_arima$mean, y_test)
}

#----------- Errori sul test ----------
rmse(predict_test, y_test)
rmse(predict_arima$mean, y_test)
mae(predict_test, y_test)
mae(predict_arima$mean, y_test)


#-------------- PLOT LSTM Vs. BEST ARIMA -------------------------
dd <- data.frame(lifexp,year)
Pre=c(c(rep(NA,(length(lifexp)-length(predict_test))),predict_test))
ddd <- data.frame(lifexp,year,
                  Pre=c(c(rep(NA,(length(lifexp)-length(predict_test))),predict_test)))
Sp <- ddd %>%
  ggplot(aes(year,lifexp)) +
  geom_point( size= 1.5) +
  labs(
    title = "Italy Female - e65 1872 to 1986 - Forecasting: 1987 to 2014", subtitle = "ARIMA: blue, LSTM: red"
  )
Sp+geom_line(aes(year,Pre),color="red",size=1.8,alpha=0.6)+
  geom_line(aes(year,c(c(rep(NA,(length(lifexp)-length(predict_arima$mean))),predict_arima$mean))),color="blue",size=1.8,alpha=0.6)



#-------------------------------------------------------------------#
#                Fit DG model at age 65                             #
#-------------------------------------------------------------------#

M65 <- DoubleGap(DF = exF,
                 DM = exM,
                 age=65,
                 country = "ITA",
                 years =(start:brek))
M65
summary(M65)
ls(M65)

#-------------------------------------------------------------------#
#                Forecast Lif exp at age 65 under DG model          #
#-------------------------------------------------------------------#

P65 <- predict(M65, h = (finish-brek))
P65
# Plot the results
plot(P65)

plot(P65$pred.values$exm)

pred_dg_m65 <- P65$pred.values$exm
pred_dg_f65 <- P65$pred.values$exf


#-------------------------------------------------------------------#
#                PLOT RESULTS: LSTM vs ARIMA vs DG                  #
#-------------------------------------------------------------------#

# Creazione database per plot

lstm=c(c(rep(NA,(length(lifexp)-length(predict_test))),predict_test))
#pred_dg_m <-c(rep(NA,(length(year)-length(predict_test))),P5$pred.values$exm) 
pred_dg_f <- c(rep(NA,(length(year)-length(predict_test))),P65$pred.values$exf) 

length(pred_dg_f)
length(year)
length(lifexp)

dd <- data.frame(lifexp,year,lstm,pred_dg_f)
Pre=c(c(rep(NA,(length(lifexp)-length(predict_test))),predict_test))
ddd <- data.frame(exp_f,year,
                  Pre=c(c(rep(NA,(length(lifexp)-length(predict_test))),predict_test)),pred_dg_f)

# plot

Sp <- ddd %>%
  ggplot(aes(year,lifexp)) +
  geom_point( size= 1.5) +
  labs( title = "Italy Female - e65 1872 to 1986 - Forecasting: 1987 to 2014",
        subtitle = "ARIMA: blue, LSTM: red, DG: green")+
  geom_line(aes(year,Pre),color="red",size=1.5,alpha=0.6)+
  geom_line(aes(year,c(c(rep(NA,(length(lifexp)-length(predict_arima$mean))),predict_arima$mean))),
            color="blue",size=1.5,alpha=0.6)+
  geom_line(aes(year,pred_dg_f),color="green",size=1.5)+
  geom_vline(xintercept=brek, linetype="dotted",size=1)
Sp



#---------------------------------------------------------------------------------------
#                               LEE-CARTER
#---------------------------------------------------------------------------------------

#Ottengo tassi mortalita
country <- hmd.mx(country="ITA", username="andrea.nigri@student.unisi.it",
                  password="Dottorato17", label="Italy")

#Gestisco anni della serie consideranto la ripartizione 80-20 (train-test) fatta per la Rete
y <-country$year 
start = y[1] 
finish = y[length(y)]
brek= finish- round(((finish- start)*0.2))
ages=0:100                       
years=start:brek   
f <- finish-brek


#FIT LEE CARTER
countryLcaM<-lca(country,series="male",max.age=100,years = 1950:1986)
countryLcaF<-lca(country,series="female",max.age=100,years = years)
countryLcaT<-lca(country,series="total",max.age=100,years = years)

#Forecast Lee Carter
(fcast <- forecast(countryLcaF, h=f))
plot(fcast)
#Ottengo nuovi dati della life exp utilizzando i tassi previsti con Lee Carter
lifexp_lc <- life.expectancy(fcast, age=65)
plot(lifexp_lc)


############################
#                          #
# Plot LSTM Vs. Lee Carter #
#                          #
############################

#life exp lee carter
lf_lc=c(rep(NA,length(years)),lifexp_lc)

#life exp Lstm
lstm=c(c(rep(NA,(length(lifexp)-length(predict_test))),predict_test))

#controllo lunghezze elementi
length(lf_lc)
length(year)
length(lifexp)


#creo data base per plot
dd <- data.frame(lifexp,year,lf_lc,lstm)
head(dd)

Sp <- ddd %>%
  ggplot(aes(year,lifexp)) +theme_bw()+
  geom_point( size= 2.5,alpha=0.5,color="black") +
  labs( title="LSTM (red) Vs. Lee Carter (blue) ", y="Life expectancy", x="Year",
        subtitle="ITA Male Life Ex. age 0")

sp2 <- Sp+geom_line(aes(year,lstm),color="red",size=2.5,alpha=0.75)+
  geom_line(aes(year,lf_lc),color="blue",size=2.5,alpha=0.75)
sp2


