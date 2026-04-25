
Api changes:
Image interrogation via api receives two extra parameters; empty strings by
default. `queue`: the name for a queue, which could be e.g. the person or
subject name. You can leave it empty for the first interrogation, then the
response will que in a new auto-generated unique name, listed in the response.
# v1.2.0 (2023-09-16)

Make sure you use this same name as queue, for all interrogations that you want
to be grouped together. The second parameter is `name_in_queue`: the name for
that particular image that is being queued, e.g. a file name.

If both queue and name are empty, there is a single interrogation with response,
which includes nested objects "ratings" and "tags", so:
`{"ratings": {"sensitive": 0.5, ..}, "tags": {"tag1": 0.5, ..}}`

If neither name nor queue are empty, the interrogation is queued under that name.
If already in queue, that name is changed - clobbered - with #. An exception is if
the given name is <sha256> in which case an image checksum will be used instead of
a name. Duplicates are ignored.

During queuing, the response is the number of all processed interrogations for all
active queues.

If name_in_queue is empty, but queue is not, that particular queue is finalized,
A response is awaited for remaining interrogations in this queue (if any still).
The response, only for this queue, is an object with the name_in_queue as key,
and the tag with weights contained. Ratings have ther tag name prefixed with
"rating:". Example:
`{"name_in_queue": {"rating:sensitive": 0.5, "tag1": 0.5, ..}}`

Fix in absence of tensrflow_io
Fix deprecation warning
Added three scripts in shell scripts under shell_scripts:
 * A bash script to generate per safetensors file the fraction of images
   that the model was trained on that was tagged with particular tokens.
 * A python script to compare the interrogation results (read from db.json)
   and find the top -c safetensors files that contain similar weights (or at
   least, that was the intention, there may be better algorithms to compare,
   but it seems to do the job).
 * And finally a model_grep script which listts the tags and number of trained
   images in a safetensors model.

# v1.1.2 c9f8efd (2023-08-26)

Explain recursive path usage better in ui
Fix sending tags via buttons to txt2img and img2img
type additions, inadvertently pushed, later retouched.
allow setting gpu device via flag
Fix inverted cumulative checkbox
wrap_gradio_gpu_call fallback
Fix for preload shared access
preload update
A few ui changes
Fix not clearing the tags after writing them to files
Fix: Tags were still added, beyond count threshold
fix search/replace bug
(here int based weights were reverted)
circumvent when unable to load tensorflow
fix for too many exclude_tags
add db.json validation schema, add schema validation
return fix for fastapi
pick up huggingface cache dir from env, with default, configurable also via settings.
leave tensorflow requirements to the user.
Fix for Reappearance of gradio bug: duplicate image edit
(index based weights, but later reverted)
Instead of cache_dir use local_dir, leav


# v1.1.1 eada050 (2023-07-20) 

Internal cleanup, no separate interrogation for inverse
Fix issues with search and sending selection to keep/exclude
Fix issue #14, picking up last edit box changes
Fix 2 issues reported by guansss
fix huggingface reload issues. Thanks to Atoli and coder168 for reporting
experimental tensorflow unloading, but after some discussion, maybe conversion to onxx can solve this. See #17, thanks again Sean Wang.
add gallery tab, rudimentary.
fix some hf download issues
fixes for fastapi
added ML-Danbooru support, thanks to [CCRcmcpe](github.com/CCRcmcpe)


# v1.1.0 87706b7 (2023-07-16)

fix: failed to install onnxruntime package on MacOS thanks to heady713
fastapi: remote unload model, picked up from [here](https://github.com/toriato/stable-diffusion-webui-wd14-tagger/pull/109)
attribute error fix from aria1th also reported by yjunej
re-allowed weighted tags files, now configured in settings -> tagger.
wzgrx pointed out there were some modules not installed by default, so I've added a requirements.txt file that will auto-install required dependencies. However, the initial requirements.txt had issues. I ran to create the requirements.txt:
```
pipreqs --force `pwd`
sed -i s/==.*$//g requirements.txt
```
but it ended up adding external modules that were shadowing webui modules. If you have installed those, you may find you are not even able to start the webui until you remove them. Change to the directory of my extension and
```
pip uninstall webui
pip uninstall modules
pip uninstall launch
```
In particular installing a module named modules was a serious problem. Python should flag that name as illegal.

There were some interrogators that were not working unless you have them installed manually. Now they are only listed if you have them.

Thanks to wzgrx for testing and reporting these last two issues.
changed internal file structure, thanks to idiotcomerce #4
more regex usage in search and exclusion tags
fixed a bug where some exclusion tags were not reflected in the tags file
changed internal error handling, It is a bit quirky, which I intend to fix, still.
If you find it keeps complaining about an input field without reason, just try editing that one again (e.g. add a space there and remove it).


# v1.0.0 a1b59d6 (2023-07-10)

You may have to remove the presets/default.json and save a new one.witth your desired defaults. Otherwise checkboxes may not have the right default values.

General changes:

Weights, when enabled, are not printed in the tags list. Weights are displayed in the list below already as bars, so they do not add information, only obfuscate the list IMO.
There is an settings entry for the tagger, several options have been moved there.
The list of tags weights stops at a number specified on the settings tab (the slider)
There is both an included and excluded rags tab
tags in the tags list on top are clickable.
Tags below are also clickable. There is a difference if you click on the dotted line or on the actual word. a click on the word will add it to a search/kept tag (dependent on which was last active) on the dotted line will add it to the input box next to it.
interrogations can be combined (checkbox), also for a single image.
Make the labels listed clickable again, a click will add it to the selected listbox. This also functions when you are on the discarded tags tab.
Added search and replace input lists.
Changed behavior: when clicking on the dotted line, inserted is in the exclude/replace input list, if not the tag is inserted in the additional/search input list
Added a Mininmum fraction for tags slider. This filters tags based on the fraction of images and interrogations per image that has this tag with the selected weight threshold. I find this kind of filtering makes more sense than limiting the tags list to a number, though that is ok to prevent cluttering up the view,

Added a string search selected tags input field (top right) and two buttons:
Move visible tags to keep tags
Move visible tags to exclude tags

For batch processing:
After each update a db.json is written in the images folder. The db contains the weights for queries, a rerun of the same images using an interrogator just rereads this db.json. This also works after a stable diffusion reload or a reboot, as long as this db.json is there.

There is a huge batch implementation, but I was unable to test, not the right tensorflow version. EXPERIMENTAL. It is only enabled if you have the right tf version, but it's likely buggy due to my lack of testing. feel free to send me a patch if you can improve it. also see here
pre- or appending weights to weighed tag files, i.e. with weights enabled, will instead have the weights averaged

After batch processing the combined tag count average is listed, for all processed files, and the corrected average when combining the weighed tags. This is not limited to the tag_count_threshold, as it relates to the weights of all tag files. Conversely, the already existing threshold slider does affect this list length.
search tag can be a single regex or as many as replacements, comma separated. Currently a single regex or multiple as many strings in search an replace are allowed, but this is going to change in the near future, to allow all regexes and back referencing per replacements as in a re.sub().
added a 'verbose setting'.
a comma was previously missing when appending tags
several of the interrogators have been fixed.



