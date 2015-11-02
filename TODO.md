# TO-DO
- FIXED. ~~BDAoOSSShape classes re-implement some of the stuff already available in BDAoOSS.bdaooss_grammar, e.g., adding, 
removing parts. I needed to do this because BDAoOSSShapeState implementation in bdaooss_grammar acts like a 
MCMCHypothesis itself, with its own implementations of likelihood, prior etc, but I needed to re-implement this and
couldn't overwrite them (since in order to overwrite them I had to change some important bits in AoMRShapeGrammar and 
BDAoOSS). Therefore, I elected to create a new BDAoOSSShape hypothesis using PCFGTree from BDAoOSS and 
BDAoOSSSpatialModel.~~
- Use log probabilities instead of probabilities to prevent any under/overflow errors.