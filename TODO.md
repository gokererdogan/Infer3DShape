# TO-DO
- FIXED. ~~BDAoOSSShape classes re-implement some of the stuff already available in BDAoOSS.bdaooss_grammar, e.g., adding, 
removing parts. I needed to do this because BDAoOSSShapeState implementation in bdaooss_grammar acts like a 
MCMCHypothesis itself, with its own implementations of likelihood, prior etc, but I needed to re-implement this and
couldn't overwrite them (since in order to overwrite them I had to change some important bits in AoMRShapeGrammar and 
BDAoOSS). Therefore, I elected to create a new BDAoOSSShape hypothesis using PCFGTree from BDAoOSS and 
BDAoOSSSpatialModel.~~
- Use log probabilities instead of probabilities to prevent any under/overflow errors.
- Split the base MCMC stuff into a separate project
- Add unit tests
- Write documentation
- You shouldn't need to traverse the whole tree every time you need to add/remove parts. We can keep a list of 
candidate nodes that is continually updated. Similarly for depth, we don't need re-calculate it every time we 
need it.
- bdaooss class depends on BDAoOSS package and that in turn depends on AoMRShapeGrammar. These packages should be 
refactored; most of it is quite messy now.
- pickle is a pain. when you change code, old pickled objects become unimportable. Implement a simpler data 
serialization functionality. A simple idea is to simply store the positions and sizes of each part, viewpoint, and
maybe params for each hypothesis. This should work for all the hypothesis in this package.
- test calculate_similarity