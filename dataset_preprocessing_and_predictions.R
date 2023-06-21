# code modified from https://github.com/JSchelldorfer/ActuarialDataScience

# setup and load data
library(CASdatasets)
library(jsonlite)

set.seed(1)

data(freMTPL2freq)

# some descriptive stats & preprocessing

? freMTPL2freq
str(freMTPL2freq)
head(freMTPL2freq, 9)

library(plyr)
# rename the Region factor levels
freMTPL2freq$Region <- revalue(
  freMTPL2freq$Region,
  c(
    "Alsace" = "R42",
    "Aquitaine" = "R72",
    "Auvergne" = "R83",
    "Basse-Normandie" = "R25",
    "Bourgogne" = "R26",
    "Bretagne" = "R53",
    "Centre" = "R24",
    "Champagne-Ardenne" = "R21",
    "Corse" = "R94",
    "Franche-Comte" = "R43",
    "Haute-Normandie" = "R23",
    "Ile-de-France" = "R11",
    "Languedoc-Roussillon" = "R91",
    "Limousin" = "R74",
    "Midi-Pyrenees" = "R73",
    "Nord-Pas-de-Calais" = "R31",
    "Pays-de-la-Loire" = "R52",
    "Picardie" = "R22",
    "Poitou-Charentes" = "R54",
    "Provence-Alpes-Cotes-D'Azur" = "R93",
    "Rhone-Alpes" = "R82"
  )
)
detach("package:plyr", unload = TRUE)

library(dplyr)

# Grouping id
distinct <- freMTPL2freq %>%
  distinct_at(vars(-c(IDpol, Exposure, ClaimNb))) %>%
  mutate(group_id = row_number())

# Preprocessing

dat <- freMTPL2freq %>%
  left_join(distinct) %>%
  mutate(
    Exposure = pmin(1, Exposure),
    Freq = pmin(15, ClaimNb / Exposure),
    VehPower = pmin(12, VehPower),
    VehAge = pmin(20, VehAge),
    # base
    # VehAge = pmin(20,VehAge^Exposure), # nonlin
    # VehAge = pmin(20, (VehAge/2)^2), # old old old nonlin...
    # VehAge = pmin(20, ceiling(VehAge^1.8/6)), # old_nonlin
    VehGas = factor(VehGas),
    DrivAge = pmin(85, DrivAge),
    logDensity = log(Density),
    VehBrand = factor(VehBrand,
      levels =
        paste0("B", c(12, 1:6, 10, 11, 13, 14))
    ),
    PolicyRegion = relevel(Region, "R24"),
    AreaCode = Area
  )

table(table(dat[, "group_id"]))
dat[dat$group_id == 283967, ] # 22 times the same row
nrow(dat)

# Covariables, Response, Exposure
x_blind <-
  c("VehPower", "VehGas", "DrivAge", "logDensity", "PolicyRegion")
x_aware <-
  c(
    "VehPower",
    "VehAge",
    "VehGas",
    "DrivAge",
    "logDensity",
    "PolicyRegion"
  )
y <- "Freq"
w <- "Exposure"

# model training and inference

library(dplyr)
library(splines)
library(splitTools)

# stratified split
ind <- partition(
  dat[["group_id"]],
  p = c(train = 0.8, test = 0.2),
  seed = 22,
  type = "grouped"
)
train <- dat[ind$train, ]
test <- dat[ind$test, ]

# fitting GLMs: using "quasipoisson" avoids warning about non-integer response.
# this has no impact on coefficients/predictions

# we start with the blind model
fit_glm_blind <- glm(
  Freq ~ VehPower +
    VehGas + ns(DrivAge, 5) + logDensity + PolicyRegion,
  data = train,
  family = quasipoisson(),
  weights = train[[w]]
)

# make predictions on test set
my_predictions_blind <-
  predict(fit_glm_blind, test[x_blind], type = "response")

# fit the aware model
fit_glm_aware <- glm(
  Freq ~ VehPower + ns(VehAge, 5) +
    VehGas + ns(DrivAge, 5) + logDensity + PolicyRegion,
  data = train,
  family = quasipoisson(),
  weights = train[[w]]
)

# make predictions on test set
my_predictions_aware <-
  predict(fit_glm_aware, test[x_aware], type = "response")

# measure performance

library(MetricsWeighted)
library(flashlight)
library(ggplot2)
fillc <- "#E69F00"

fl_glm_blind <- flashlight(
  model = fit_glm_blind,
  label = "GLM blind",
  predict_function = function(fit, X) {
    predict(fit, X, type = "response")
  }
)

fl_glm_aware <- flashlight(
  model = fit_glm_aware,
  label = "GLM aware",
  predict_function = function(fit, X) {
    predict(fit, X, type = "response")
  }
)


# Combine them and add common elements like reference data
metrics <-
  list(
    `Average deviance` = deviance_poisson,
    `Relative deviance reduction` = r_squared_poisson
  )
fls <-
  multiflashlight(
    list(fl_glm_blind, fl_glm_aware),
    data = test,
    y = y,
    w = w,
    metrics = metrics
  )

# Version on canonical scale
fls_log <- multiflashlight(fls, linkinv = log)

perf <- light_performance(fls)
perf
plot(perf, geom = "point") +
  labs(x = element_blank(), y = element_blank())

# save data

write.csv(freMTPL2freq, "Data/freMTPL2freq.csv")
write_json(x_aware, "Data/x_aware.json")
write_json(x_blind, "Data/x_blind.json")
write_json(y, "Data/y.json")
write_json(ind, "Data/ind.json")
write.csv(test, "Data/freMTPL2freq_preprocessed_test.csv")
write.csv(train, "Data/freMTPL2freq_preprocessed_train.csv")
write_json(my_predictions_aware, "Data/my_predictions_aware.json")
write_json(my_predictions_blind, "Data/my_predictions_blind.json")
