# test_redo_k_11

These scripts re-do some of the scripts in the main folder, but with the blending midpoint lambda_k set to 100.  This is for better comparison with Target-Thresh and Target+Count encodings.  

Encoded features are also saved with standard naming conventions for ease of use later.

Code runs using the instructions in the parent folder.  After running data imports, the scripts in this folder can be run.

In addition to code here, 14_xgb_menc_loop_k.ipynb was added to the parent folder to get standard mean encoding at various lambda_k (including 100)