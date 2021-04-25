


library(lattice)
library(ggplot2)


setwd('P:/AF_Project/Data/')
mydf = read.csv('Sub_AF_Data.csv')
rm_obs = read.csv('Discharge_Diagnosis.csv')
mydf = mydf[which(rm_obs$D_DIAG == 0),]

mydf$y_AF[which(mydf$y_AF == 1)] = 'AF'
mydf$y_AF[which(mydf$y_AF == 0)] = 'non-AF'



tmpdf = mydf[,c('y_AF', 'ECG_PR')]
tmpdf = na.omit(tmpdf)
colnames(tmpdf) = c('y', 'x')
tmpdf$y = as.factor(tmpdf$y)
lbd = quantile(tmpdf$x, 0.001)
ubd = quantile(tmpdf$x, 0.95)
tmpdf = tmpdf[which(tmpdf$x > lbd), ]
tmpdf = tmpdf[which(tmpdf$x < 300), ]

ggplot(tmpdf, aes(y, x, fill = y, color = y))+
  geom_violin()+
  geom_boxplot(width = 0.1, lwd = 2.5, fill = 'cyan1', color = 'darkcyan')+
  ylim(120, 350)


tmpdf = mydf[,c('y_AF', 'AGE')]
tmpdf = na.omit(tmpdf)
colnames(tmpdf) = c('y', 'x')
tmpdf$y = as.factor(tmpdf$y)
ggplot(tmpdf, aes(y, x, fill = y, color = y))+
  geom_violin()+
  geom_boxplot(width = 0.1, lwd = 2.5, fill = 'cyan1', color = 'darkcyan')+
  ylim(20, 120)



tmpdf = mydf[,c('y_AF', 'UCC_TTE_LAD')]
tmpdf = na.omit(tmpdf)
colnames(tmpdf) = c('y', 'x')
tmpdf$y = as.factor(tmpdf$y)
lbd = quantile(tmpdf$x, 0.001)
ubd = quantile(tmpdf$x, 0.999)
tmpdf = tmpdf[which(tmpdf$x > lbd), ]
tmpdf = tmpdf[which(tmpdf$x < ubd), ]
ggplot(tmpdf, aes(y, x, fill = y, color = y))+
  geom_violin()+
  geom_boxplot(width = 0.1, lwd = 2.5, fill = 'cyan1', color = 'darkcyan')+
  ylim(20, 80)



tmpdf = mydf[,c('y_AF', 'ECG_HR')]
tmpdf = na.omit(tmpdf)
colnames(tmpdf) = c('y', 'x')
tmpdf$y = as.factor(tmpdf$y)
lbd = quantile(tmpdf$x, 0.001)
ubd = quantile(tmpdf$x, 0.999)
tmpdf = tmpdf[which(tmpdf$x > lbd), ]
tmpdf = tmpdf[which(tmpdf$x < ubd), ]
ggplot(tmpdf, aes(y, x, fill = y, color = y))+
  geom_violin()+
  geom_boxplot(width = 0.1, lwd = 2.5, fill = 'cyan1', color = 'darkcyan')+
  ylim(40, 185)


setwd('P:/AF_Project/Data/')
mydf = read.csv('Sub_AF_Data.csv')
rm_obs = read.csv('Discharge_Diagnosis.csv')
mydf = mydf[which(rm_obs$D_DIAG == 0),]
mydf = mydf[which(mydf$IMG_IS_CII != 0),]
#mydf = mydf[which(mydf$IMG_IS_CII == 1),]
#mydf = mydf[which(mydf$IMG_IS_CII == 2),]
#dim(mydf)
mydf$y_AF[which(mydf$y_AF == 1)] = 'AF'
mydf$y_AF[which(mydf$y_AF == 0)] = 'non-AF'



tmpdf = mydf[,c('y_AF', 'IMG_IS_CII')]
tmpdf[is.na(tmpdf$IMG_IS_CII), 2] = 0
colnames(tmpdf) = c('AF', 'IMG_IS_CII')
tmpdf$AF = as.factor(tmpdf$AF)
tmpdf$IMG_IS_CII[tmpdf$IMG_IS_CII == 1] = 'Yes'
tmpdf$IMG_IS_CII[tmpdf$IMG_IS_CII == 2] = 'No'

ggplot(tmpdf, aes(AF, fill = IMG_IS_CII)) +
  geom_bar(width = 0.5, position = 'fill', alpha = 0.75) +
  labs(y = 'Percentage') +
  ylim(0, 1.3)

ggplot(tmpdf, aes(IMG_IS_CII, fill = AF)) +
  geom_bar(width = 0.5, position = 'dodge')

