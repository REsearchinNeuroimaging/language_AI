lan_AI: AI based language organisation recognition
=======================================================

"lan_AI" is a tool for recognising atypical or typical language organisation in resting-state functional magnetic resonance imaging (RS-fMRI) data. The program takes a preprocessed 4D NIfTI file as input and outputs the predicted probability of atypical language organization. 

This tool should be used only **after** BOLD data are preprocessed and normalized to the  *MNI152NLin2009cAsym* template (for example after preprocessing the data using ``fmriprep``).

Running
----------

To run the script, you need to place the "language_AI.py" file and the "requirements" directory along with all of its contents in a common directory.

Command

``python language_AI.py <path_to_nifti_file> <tr>``
