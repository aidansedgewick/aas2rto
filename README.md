# aas2rto

Tools to prioritize targets for follow up, based (currently!) on ZTF and ATLAS datastreams.
The concept is to reduce a target down to a single "score" by which targets can be ranked.
You can write your own `scoring_function`to prioritise them as you like, and `modeling_function` to use more than just the lightcurve data.

aas2rto is written in python, using >=3.9 is recommended (<3.8 is not guaranteed to work).
  
## Installing
To install
- Clone this repo: `git clone https://github.com/aidansedgewick/aas2rto.git`
- Preferably start a new virtualenv: `python3 -m virtualenv env_aas2rto`
    - You may need to `python3 -m pip install virtualenv` 
    - Then `source env_aas2rto/bin/activate` - do this every session!
- Move to the cloned directory: `cd aas2rto`
- Install requirements `python3 -m pip install -r requirements.txt`
- Install this package: `python3 -m pip install -e .` 
    - ideally install as developer -this is the `-e` flag
- Optionally run the tests: `pytest test_aas2rto --cov aas2rto`

## Quickstart
There are some example configs:

#### `config/examples/fink_supernovae.yaml` :
- Modify this file with your FINK and ATLAS credentials
    - If you do not have ATLAS credentials, set `use: False` in the ATLAS config section.
- use this configuration with:
     `python3 scripts/main.py -c config/fink_supernovae.yaml -x`
    - (the `-x` flag recovers existing targets if the program crashes...)
- watch the terminal output and watch the folder `fink_supernovae/outputs/plots` for new plots...
    
#### `config/examples/fink_kn_bot.yaml`:
- Listen for ZTF alerts classified as kilonovae, and send alerts though telegram! 
- Add your credentials 
    - As earlier, modify the config and switch ATLAS `use: False` if you don't have the credentials yet.
- If you do not already have a suitable telegram bot set up, there is a brief description below.
- Run with `python3 scripts/main.py --config config/fink_kn_bot.yaml`

#### `config/examples/alerce_supernovae.yaml`:
Note that there are no credentials needed for Alerce!
- Alerce doesn't have kafka alerts at the moment, so this script periodically queries for new targets.
- Add Atlas credential, or switch off as earlier.
- run with `python3 scripts/main.py --config config/alerce_supernovae.yaml`

#### `config/examples/yse_rising.yaml`:
- Add credentials as before.
NOTE: the YSE module is still experimental...!

## The main interface
Once the quickstart isn't much use to you any more, you can write your own scoring functions and modeling functions.

There is a "blank" config in `config`. You can edit this, add whatever parameters you like. Note that all the
query_managers are turned to `use: False` by default.

Then: `python3 scripts/main.py`

## Scoring functions

The 'science' scoring function encodes how interesting a target is.

If you write your own scoring function:
it should accept two arguments:
    target: a `dk154_targets.target.Target` object.
    t_ref: an `astropy.time.Time`

it should return either:
    score (`float`)
    OR
    score (`float`), scoring_comments (`List of str`)

notes about score:
    if `score` is +ve: The target will be ranked according to the score
    if `score` is -ve: The target will be excluded from the ranked lists, but not removed.
    if `score` is -np.inf: the target will be rejected, and removed from the program.


An example is given here:

```
def my_science_scoring_function(target, t_ref):

    #=====Keep track of some stuff.
    scoring_comments = []
    reject_comments = []
    factors = []
    reject = False    
    exclude = False

    #=====How bright?

    fink_data = target.target_data.get("fink", None)


    mag = fink_data.detections["magpsf"].values[-1]
    mag_factor = 10 ** (18.0 -mag) # bright targets have high mag
    factors.append(mag_factor)
    scoring_comments.append(f"mag_factor={mag_factor}")
    if mag > 22:
        reject_comments.append(f"REJECT: {mag} is too faint!")
        reject = True        

    #=====How many obs?
    num_obs_factor = len(fink_data.detections)
    # more observations are more reliable!
    factors.append(num_obs_factor)
    
    model = target.models.get("complicated_model", None)
    if model is not None:
        chi2 = model.chi2
        model_factor = 1. / chi2
        # High chi2 is bad!
        scoring_comments.append(f"chi2 factor ={model_factor} from {chi2:.3f}")

    # Final score
    score = target.base_score # probably 100
    for fac in factors:
        score = score * fac
    
    if exclude is True:
        score = -1.0 # Not interesting now, but may be later/other observatories.

    if reject is True:
        score = -np.inf # Will never be interesting.

    scoring_comments.extend(reject_comments)
    
    return score, scoring_comments
```

You should then pass `my_science_scoring_function` to `TargetSelector.start()` :
```
from dk154_targets import TargetSelector
from dk154_targets.modeling import sncosmo_model

from my_file import my_scoring_function

selector = TargetSelector.from_config("/path/to/config/")
selector.start(scoring_function=my_science_scoring_function)
```

(see main.py for details)

This function also uses a model, generated by a function called `compicated_model`. If you want to use models 
in scoring your targets, you should also pass the modeling_function to `TargetSelector.start()`.
The model will be stored in `target.models[function_name]` (see below for more details).


### observatory scoring function

You probably also want to factor in how visible a target is from a given observatory.

You can provide a separate scoring function for this.
The function must have signature:
    target : aas2rto.target.Target
    observatory : astroplan.Observer
    t_ref : astropy.time.Time

As it can be expensive to compute, some information about each target at an observatory is computed in advance of scoring. 
It is saved in a dictionary in `target.observatory_info`
eg. `example_targ.observatory_info["lasilla"]`.
Each entry is an `aas2rto.obs_info.ObservatoryInfo`.

eg. 

```
def my_obs_scoring(target, obseratory, t_ref):

    exclude = False
    comments = []

    obs_name = observatory.name
    obs_info = target.obseravtory_info[obs_name]

    # find closest pre-computed altitude.
    mask = obs_info.t_grid.mjd < t_ref.mjd # all times in t_grid before now
    latest_altaz = obs_info.target_altaz[mask][-1] # array of astropy.AltAz

    alt = latest_altaz.alt

    if alt.deg < 30.0:
        exclude = True
        comments.append(f"EXCLUDE: alt {alt.deg:.1f}deg less than required 30deg")

    airmass = 1 / np.sin(alt.rad)
    obs_score = 1 / airmass

    if exclude:
        obs_score = -1.0 # so that we can exclude 

    return obs_score, comments
```

You should pass your new function to the selector at `start()`:
```
ts = TargetSelector.from_config(...)
ts.start(
    scoring_function=my_science_scoring_function,
    observatory_scoring_function=my_obs_scoring,
)
```

If the target has a valid (positive) score from `my_science_scoring_function`,
the extra factors from `my_obs_scoring` will be included.

There are default observatory_scoring_functions in 
`aas2rto.scoring.default_obs_scoring`.


### classes as Scoring functions
A fancier way to write a scoring function would be to write a `class` with the `__call__` magic method.
This could be helpful if there are some parameters that your scoring function depends on, that you might like to change easily (without editing all the code for your function).
NB: your class **must** implement the `__name__` attribute in init.
eg
```
class MyScoringFunction:

    def __init__(self, mag_lim=18.5):
        self.__name__ = "my_scoring_function"  # Definitely remember to do this!!
        self.mag_lim = mag_lim
        __name__ = "my_scoring_function"        

    def __call__(self, target: Target, observatory: Observer, t_ref: Time):
        exclude = False
        last_mag = target.fink_data.detections["magpsf"].values[-1]
        if last_mag > self.mag_lim:
            exclude = True # Exclude faint targets!
        if exclude:
            return -1.0
        return 100.0
```

Then remember to initialise the class before you pass it to a `TargetSelector`!
```
from dk154_targets import TargetSelector
from dk154_targets.modeling import sncosmo_model

from my_file import MyScoringFunction

scoring_function = my_scoring_function(mag_lim=20.5) # Actually, fainter targets are ok!

selector = TargetSelector.from_config("/path/to/config/")
selector.start(
    scoring_function=scoring_function,
    modeling_function=sncosmo_model
)
```

## Modeling functions

You might want to build models for your targets to base the score around.

You can write a modeling function to build a model based on the target - your function should accept a single argument, `target` (which will be a `Target`)

```
from blah import ComplicatedModel

def complicated_model(target):
    mag_data =  target.alerce_data.lightcurve["magpsf"]
    jd_data = target.alerce_data.lightcurve["jd"]
    model = ComplicatedModel(jd_data, mag_data)
    return model
```

You then give this function to `TargetSelector.start()`

The model is saved in the target in a dictionary `target.models`, with the key as the name of the function,
`my_model = target.models["complicated_model"]`

NB: if the model fitting fails (an exception is raised by your function), then `None` will be stored in the models dictionary. Build this into your scoring function accordingly! (ie, you may want to reject the target if the model failed, or just not take this information into account.)

### Want to fit more than one model?
pass a list of functions,
`selector.start(modeling_function=[some_model, another_model])`

Then:
`m1 = target.models.get("some_model")` and `m2 = target.models.get("another_model")`

## Telegram bot


## A few useful scripts...
- `scripts/get_atlas_token.py`
    - prints your token to the screen.
