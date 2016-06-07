# TO-DO
- FIXED. ~~BDAoOSSShape classes re-implement some of the stuff already available in BDAoOSS.bdaooss_grammar, e.g., adding, 
removing parts. I needed to do this because BDAoOSSShapeState implementation in bdaooss_grammar acts like a 
MCMCHypothesis itself, with its own implementations of likelihood, prior etc, but I needed to re-implement this and
couldn't overwrite them (since in order to overwrite them I had to change some important bits in AoMRShapeGrammar and 
BDAoOSS). Therefore, I elected to create a new BDAoOSSShape hypothesis using PCFGTree from BDAoOSS and 
BDAoOSSSpatialModel.~~
- ~~Use log probabilities instead of probabilities to prevent any under/overflow errors.~~
- ~~Split the base MCMC stuff into a separate project~~
- ~~Add unit tests~~
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
- viewpoint and camera position is confounded now. if viewpoint is available, camera_pos setting in 
vision_forward_model is ignored. but this is confusing. viewpoint refers to the rotation of the object around z axis. 
camera_pos is the set of positions of the camera. One should be able to change these independently. Let viewpoint and 
camera_position be both rotation angles (position in spherical coordinates). when rendering, add the viewpoint and
camera_pos angles together. If no viewpoint, assume 0 rotation for object.
- Don't change object internal stuff from the outside. Most proposal functions do that. The class should take care 
of its internal stuff itself. Changing from the outside makes it harder to control consistency of the internal 
information. For example, proposal functions add/remove voxels directly by accessing internal stuff; this could 
lead to cases where hypothesis depth is out of sync. The best way is to let objects take care of their internals
themselves. This is also important because we assume hypotheses don't change; prior and likelihood is calculated
only once. Therefore, if we make any changes in hypothesis, we force a recalculation of these. This is ugly and
potentially dangerous. The solution is to make sure that all the info you need is already final before you create
a hypothesis. Your proposal functions should create all the internal info first and then create the 
hypothesis using these.
- ~~add resize voxel move to voxel based shape. that should enable the coarsest level to settle down on the extent of the
object, hopefully enabling faster convergence.~~
- do not find the list of partial, full, empty voxels from scratch each time. keep a list of these at the highest level 
and update these as we add/remove/change voxels.
- ~~write tests for voxel_based_shape scale and origin.~~
- write tests for voxel count_trees
- ~~We shouldn't let part size to get really small (around 0.01 in any direction for example). Such small parts do not make
sense. The right way to do this is by changing the prior on part sizes. Maybe we should constrain the range from below at
0.01.~~
- ~~Use spherical coordinates for viewpoint. Convert them to cartesian in vision_forward_model~~
- You can't have multiple offscreen rendering forward models in VTK. See if there is a way around this.
- Reformat docstrings so sphinx can autogenerate documentation. Should I use native sphinx format or numpydoc?
- BDAoOSSShapeMaxD (also ShapeMaxN) should constrain the max depth themselves; right now proposal functions ensure that.
This problem will be solved if we do not modify the insides of these from the proposal functions. Let these functions
handle their internals themselves.
- Get rid of BDAoOSSShapeMaxD because it is using the same prior with BDAoOSSShape.
- A much cleaner way is for the add/remove part proposals to ask the shape instance if we can add/remove a part, and 
also return where we can add them too. 
- Proposal function still needs to know what the maximum number of parts is. Maybe we can define maxn in all classes,
and let this be inf if there is no limit. 
- Unittest for paperclip_shape.calculate_moment_of_inertia is missing.
- vision_forward_model RGB rendering tests
- vision_forward_model.convert_world_to_display test