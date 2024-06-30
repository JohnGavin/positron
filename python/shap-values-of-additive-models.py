# https://lorentzen.ch/index.php/2024/06/28/shap-values-of-additive-models/

# https://github.com/pyenv/pyenv/blob/master/COMMANDS.md
# pyenv --help
# pyenv init --path
# pyenv version # 3.10.9
# which python is activated (in pyenv) where * is placed
# !pyenv versions 
# local there are global and shell levels too. 
#   shell overrides local and global, 
#   local overrides global
# set locally which python in pyenv to use
# !pyenv global 3.10.9
# !pyenv local 3.10.9
# path to which python is activated in pyenv 
# !pyenv which python 
# list packages 
# !pyenv exec pip list
# install pkg1 pkg2 then repeat list command above
# !pyenv exec pip install lightgbm shap
# run helloworld.py using pyenv
# !pyenv exec python helloworld.py

system.time({  # 1s
  shap_lm <- list(
    add = shapviz(additive_shap(fit_lm, df)),
    kern = kernelshap(fit_lm, X = df[xvars], bg_X = df),
    perm = permshap(fit_lm, X = df[xvars], bg_X = df)
  )

  shap_lgb <- list(
    tree = shapviz(fit_lgb, X),
    kern = kernelshap(fit_lgb, X = X, bg_X = X),
    perm = permshap(fit_lgb, X = X, bg_X = X)
  )
})

# Consistent SHAP values for linear regression
all.equal(shap_lm$add$S, shap_lm$perm$S)
all.equal(shap_lm$kern$S, shap_lm$perm$S)

# Consistent SHAP values for boosted trees
all.equal(shap_lgb$lgb_tree$S, shap_lgb$lgb_perm$S)
all.equal(shap_lgb$lgb_kern$S, shap_lgb$lgb_perm$S)

# Linear coefficient of x3 equals slope of SHAP values
tail(coef(fit_lm), 1)                # 0.682815
diff(range(shap_lm$kern$S[, "x3"]))  # 0.682815

sv_dependence(shap_lm$add, xvars)
sv_dependence(shap_lm$add, xvars, color_var = NULL)



# !pip3 install lightgbm shap sklearn-learn
import numpy as np
import lightgbm as lgb
import shap
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

#===================================================================
# Make small data
#===================================================================

def make_data(n=100):
    x1 = np.linspace(0.01, 1, n)
    x2 = np.log(x1)
    x3 = x1 > 0.7
    X = np.column_stack((x1, x2, x3))

    y = 1 + 0.2 * x1 + 0.5 * x2 + x3 + np.sin(2 * np.pi * x1)
    
    return X, y

X, y = make_data()

#===================================================================
# Additive linear model and additive boosted trees
#===================================================================

# Linear model with polynomial terms
poly = PolynomialFeatures(degree=3, include_bias=False)

preprocessor =  ColumnTransformer(
    transformers=[
        ("poly0", poly, [0]),
        ("poly1", poly, [1]),
        ("other", "passthrough", [2]),
    ]
)

model_lm = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("lm", LinearRegression()),
    ]
)
_ = model_lm.fit(X, y)

# Boosted trees with single-split trees
params = dict(
    learning_rate=0.05,
    objective="mse",
    max_depth=1,
    colsample_bynode=0.7,
)

model_lgb = lgb.train(
    params=params,
    train_set=lgb.Dataset(X, label=y),
    num_boost_round=300,
)