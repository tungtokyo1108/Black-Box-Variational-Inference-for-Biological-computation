# this data set was analyzed in Zhao 2011 (Nature Communications 2:467)
line <- read.csv("RiceDiversityLine.csv")
pheno <- read.csv("RiceDiversityPheno.csv")
geno <- read.csv("RiceDiversityGeno.csv")
line.pheno <- merge(line, pheno, by.x = "NSFTV.ID", by.y = "NSFTVID")
alldata <- merge(line.pheno, geno, by.x = "NSFTV.ID", by.y = "NSFTVID")

mydata <- data.frame(
  # Flowering time 
  flower.Aber = alldata$Flowering.time.at.Aberdeen,
  flower.Ark = alldata$Flowering.time.at.Arkansas,
  flower.Fari = alldata$Flowering.time.at.Faridpur,
  # Morphology
  culm = alldata$Culm.habit,
  leaf.length = alldata$Flag.leaf.length,
  leaf.width  = alldata$Flag.leaf.width,
  # Yeild components
  plant.height = alldata$Plant.height,
  panicle.length = alldata$Panicle.length,
  pri.panicle.branch = alldata$Primary.panicle.branch.number,
  seed.panicle = alldata$Seed.number.per.panicle,
  flor.panicle = alldata$Florets.per.panicle,
  panicle.fertility = alldata$Panicle.fertility,
  # Seed morphology
  seed.length = alldata$Seed.length,
  seed.width = alldata$Seed.width,
  seed.volum = alldata$Seed.volume,
  seed.surface = alldata$Seed.surface.area,
  brown.length = alldata$Brown.rice.seed.length,
  brown.width = alldata$Brown.rice.seed.width,
  brown.surface = alldata$Brown.rice.surface.area,
  brown.volume = alldata$Brown.rice.volume,
  # Stress tolerance
  straighhead = alldata$Straighthead.suseptability,
  blast = alldata$Blast.resistance,
  # Quality 
  amylose = alldata$Amylose.content,
  alkali.spreading = alldata$Alkali.spreading.value,
  protein = alldata$Protein.content
)
missing <- apply(is.na(mydata), 1, sum) > 0
mydata <- mydata[!missing, ]

# Compute PCA 
res <- prcomp(mydata, scale = T)
summary(res)
plot(res)

subpop <- alldata$Sub.population[!missing]
op <- par(mfrow=c(1,2))
plot(res$x[,1:2], col=as.numeric(subpop))
plot(res$x[,3:4], col=as.numeric(subpop))
par(op)

library(ggbiplot)
op <- par(mfrow=c(1,2))
ggbiplot(res, obs.scale = 1, var.scale = 1,
         groups = subpop, ellipse = TRUE, circle = TRUE) +
  scale_color_discrete(name = '') +
  theme(legend.direction = 'horizontal', legend.position = 'top')
ggbiplot(res, obs.scale = 1, var.scale = 1, choices = c(3,4),
         groups = subpop, ellipse = TRUE, circle = TRUE) +
  scale_color_discrete(name = '') +
  theme(legend.direction = 'horizontal', legend.position = 'top')
par(op)

# Visualization for PCA 
library("factoextra")
library("corrplot")
fviz_eig(res, addlabels = TRUE, ylim = c(0, 35))

# Result for variables 
res.var <- get_pca_var(res)
res.var$cor    # Correlations between variables and dimensions
corrplot(res.var$cor, is.corr=FALSE)
op <- par(mfrow = c(1,2))
fviz_pca_var(res, axes = c(1,2), col.var = "black", repel = TRUE)
fviz_pca_var(res, axes = c(3,4), col.var = "black", repel = TRUE)
par(op)
fviz_pca_var(res, axes = c(5,6), col.var = "black", repel = TRUE)

# Coordinates of variabels to create a scatter plot
res.var$coord   
# Formula for Cooridinates 
var_coord_func <- function(loadings, comp.sdev) {
  loadings * comp.sdev
}
loadings <- res$rotation
sdev <- res$sdev
res.var.coord <- t(apply(loadings, 1, var_coord_func, sdev))

# - Represents the quality of representation for variables on the factor map
# - Indicate the contribution of a component to the squared distance of 
#   the observation to the origin
# - Components with a large value of cos2 contribute a relatively large portion to
#   the total distance and therefore these components are importance for that observation 
# - The closer a variable is to the circle of correlation, 
#   the better its representation on the factor map (the more important it is to interpret components)
res.var$cos2  
# Formula for Cos2 
res.var.cos2 <- res.var.coord^2
fviz_pca_var(res, axes = c(1,2), col.var = "cos2", gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE)
fviz_pca_var(res, axes = c(3,4), col.var = "cos2", gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE)


# The contributions of the variables in accounting for the variability in 
# a given principal component are expressed in percentage 
# - Variables that are correlated with PC1 and PC2 are the most important in explaining 
#   the variability in the data set 
# - Variables that do not correlated with any PC or correlated with the last dimension 
#   are variables with low contribution and migh be removed to simplify the overall analysis
res.var$contrib 
fviz_contrib(res, choice = "var", axes = 1)
fviz_contrib(res, choice = "var", axes = 2)
# The red dashed line on the graph above indices the expected average contribution. 
# If contribution of the variables were uniform, the expected value would be 
# 1/length(variables) = 1/20.
fviz_pca_var(res, col.var = "contrib", gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE)

ind <- get_pca_ind(res)
fviz_pca_ind(res, geom.ind = "point", col.ind = subpop,
             addEllipses = TRUE, axes = c(1,2), ellipse.type = "confidence",
             legend.title = "Sub-population")
fviz_pca_ind(res, geom.ind = "point", col.ind = subpop, 
             addEllipses = TRUE, axes = c(3,4), ellipse.type = "confidence",
             legend.title = "Sub-population")

# Bitplot
fviz_pca_biplot(res, axes = c(1,2), 
                # Individuals
                geom.ind = "point", 
                fill.ind = subpop, 
                pointshape = 21, pointsize = 2, 
                addEllipses = TRUE, ellipse.type = "confidence",
                # Varibles
                alpha.var = "contrib", col.var = "contrib",
                gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), repel = TRUE,
                legend.title = list(fill = "Sub-population", color = "Contrib",
                                    alpha = "Contrib"))

fviz_pca_biplot(res, axes = c(3,4), 
                # Individuals
                geom.ind = "point", 
                fill.ind = subpop, 
                pointshape = 21, pointsize = 2, 
                addEllipses = TRUE, ellipse.type = "confidence",
                # Varibles
                col.var = "contrib",alpha.var = "contrib",
                gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), repel = TRUE,
                legend.title = list(fill = "Sub-population", color = "Contrib",
                                    alpha = "Contrib"))
