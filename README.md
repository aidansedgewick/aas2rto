# dk154-targets-py38

Tools to prioritize targets for follow up, based (currently!) on ZTF and ATLAS datastreams.
The concept is to reduce a target down to a single "score" by which targets can be ranked.
You can write your own `scoring_function`to prioritise them as you like, and `modeling_function` to use more than just the 

  
## Installing
To install
- Clone this repo: `git clone https://github.com/aidansedgewick/dk154-targets-py38.git`
- Preferably start a new virtualenv: `python3.8 -m virtualenv dk154_env`
    - You may need to `python3.8 -m pip install virtualenv` 
    - Then `source dk154_env/bin/activate` - do this every session!
- Move to the cloned directory: `cd dk154-targets-py38`
- Install requirements `python3 -m pip install -r requirements.txt`
- Install this package: `python3 -m pip install -e .` 
    - ideally install as developer -this is the `-e` flag
- Optionally run the tests: `pytest --cov dk154_targets`

## Quickstart
There are three example configs:

### `config/examples/fink_supernovae.yaml` :
- Modify this file with your FINK and ATLAS credentials
    - If you do not have ATLAS credentials, set `use: False` in the ATLAS config section.
- use this configuration with:
     `python3 scripts/main.py -c config/fink_supernovae.yaml --existing read`
    - (the `--existing read` flag recovers existing targets if the program crashes...)
- watch the terminal output and watch the folder `fink_supernovae/outputs/plots` for new plots...
    
### `config/examples/fink_kn_bot.yaml`:
- Listen for ZTF alerts classified as kilonovae, and send alerts though telegram! 
- Add your credentials 
    - As earlier, modify the config and switch ATLAS `use: False` if you don't have the credentials yet.
- If you do not already have a suitable telegram bot set up, there is a brief description below.
- Run with `python3 scripts/main.py -c config/fink_kn_bot.yaml`

### `config/examples/alerce_supernovae.yaml`:
Note that there are no credentials needed for Alerce!
- Alerce doesn't have kafka alerts at the moment, so this script periodically queries for new targets.
- Add Atlas credential, or switch off as earlier.
- run with `python3 scripts/main.py -c config/alerce_supernovae.yaml`

### `config/examples/yse_rising.yaml`:
- Add credentials as before.
NOTE: the YSE interface is still experimental...!

## The main interface
Once the quickstart isn't much use to you any more, you can write your own scoring functions and modeling functions.

There is a "blank" config in `config`. You can edit this, add whatever parameters you like. Note that all the
query_managers are turned to `use: False` by default.

Then: `python3 scripts/main.py`

## Scoring functions

If you write your own scoring function:
it should accept three arguments:
    target: a `dk154_targets.target.Target` object.
    observatory: an `astroplan.Observer`.
    t_ref: an `astropy.time.Time`

it should return either:
    score: a single float
    OR
    score, scoring_comments, reject_comments

notes about score:
    if `score` is +ve: The target will be ranked according to the score
    if `score` is -ve: The target will be excluded from the ranked lists, but not removed.
    if `score` is -np.inf: the target will be rejected, and removed from the program.


it should look something like:

```
def my_scoring_function(target, observatory, t_ref):

    # Keep track of some stuff.
    scoring_comments = []
    reject_comments = []
    factors = []
    reject = False    
    exclude = False

    # How bright?
    mag = target.fink_data.detections["magpsf"].values[-1]
    mag_factor = 10 ** (18.0 -mag) # bright targets have high mag
    factors.append(mag_factor)
    scoring_comments.append(f"mag_factor={mag_factor}")
    if mag > 22:
        reject_comments.append(f"{mag} is too faint!")
        reject = True        

    # How many obs?
    num_obs_factor = len(target.fink_data.detections)
    # more observations are more reliable!
    factors.append(num_obs_factor)
    
    model = target.models.get("complicated_model", None)
    if model is not None:
        chi2 = model.chi2
        model_factor = 1. / chi2
        # High chi2 is bad!
        scoring_comments.append(f"chi2 factor ={model_factor} from {chi2:.3f}")
        
    if observatory is not None:
        obs_name = getattr(observatory, "name", None)
        observing_info = target.observatory_info.get(obs_name)

        target_altaz = observing_info.target_altaz
        if target_altaz is not None:
            if all(target_altaz.alt.deg<30.):
                exclude = True

    # Final score
    score = target.base_score # probably 100
    for fac in factors:
        score = score * fac
    
    if exclude is True:
        score = -1.0 # Not interesting now, but may be later/other observatories.

    if reject is True:
        score = -np.inf # Will never be interesting.
    
    return score, scoring_comments, reject_comments
```

You should then pass `my_scoring_function` to `TargetSelector.start()` (see main.py for details)

This function also uses a model, generated by a function called `compicated_model`. If you want to use models 
in scoring your targets, you should also pass the modeling_function to `TargetSelector.start()`.
The model will be stored in `target.models[function_name]`.

More details soon...

## A few useful scripts...
- `scripts/get_atlas_token.py`
    - prints your token to the screen.

