# Multiview Modeler — Blender add-on

Builds a dense mesh of a character or object from multi-view maps. Workflow:

1. Start volume from all views (`normals_volume.py`).
2. Fit the mesh to all maps at once (`multiview_fit.py`): protection against flipped faces, mesh relaxation after every step.
3. Reduce to a polygon limit.

## Installation

1. Edit → Preferences → Add-ons → Install… → pick `multiview_modeler.zip` and enable "Multiview Modeler". In Blender 4.2+ this is Add-ons → ▾ → *Install from Disk*.
2. Sidebar (N) → **Multiview** tab → **Install dependencies**. Installs scipy, scikit-image and pillow once, about 150 MB. They go to a separate folder `scripts/mvm_deps` and are used only by the background process. Blender's own Python is not modified.

## Workflow

1. **Views → 8 standard views**. For every view set a **Normal map** or a **Measurement view**, then press **Normals from measurement views**. Each view has an enable flag, a manual X/Y shift and a normal-map format.
   The list of views can have any number of entries: baking fills it from the files, and the **+** / **−** buttons add or remove views by hand.
2. **Height** — character height in scene units.
3. **Start live preview** — a background process loads the maps once and integrates the depth (about 20 s). A progress bar under the button shows the stage and percentage (the same bar is used during the final build), and **Stop live preview** ends the process. After that every slider move rebuilds the `MVM_Preview` mesh after the **Update delay**:
   - volume sliders take a few seconds;
   - every preview fit iteration takes a few more seconds.
4. When the preview looks right, press **Final build**: the same settings at full resolution. The result is `MVM_Result`, the start mesh is `MVM_Underlay`.
5. **Game mesh → Reduce to limit** — Decimate towards the limit. The mesh is checked before and after (open edges, non-manifold edges, separate parts); if decimation would open holes or split parts, the ratio is raised and the limit is exceeded (turn off **Allow exceeding limit** to force the limit). The result is reported. A full copy is kept for baking.

### Panel layout

The top holds the common settings and the method choice (**Method**). Below it there is one collapsible section per method. The active method is marked with a dot in its header. Each method section ends with a **Reset method settings** button.

1. **Volume from views** — normal maps only. Max carve depth, Carve thin above, Thin, Smoothing; "Fit to silhouettes" and "Fit to normal maps" blocks.
2. **Hunyuan3D start** — colour view + normal maps. Install, generate, the same two blocks.
3. **Depth + normals (C2)** — a depth map for every view. Grid, Smoothing (off = C2-B), Auto depth offset, PNG format, optional extra fit.

Controls that did not improve results in tests were removed and fixed to the tested values:
- silhouette rim 3 px, view blend softness 0.8, cut at normal jump 0.6, smoothing threshold 0.004;
- flip protection 0.3 → 0.1, step 0.05, view agreement 25°;
- silhouette fit step 0.8, silhouette resolution 512.

**Display → Highlight** shows where the mesh disagrees with the maps (*Disagreement with views*) and what no view sees (*Guessed areas*). **Reset** restores the tested values.

Two checkboxes in the main panel clean the result in both the live preview and the final build:
- **Trim outside** removes geometry that sticks out of the silhouettes by more than 2 % of the height (for example the background a generator leaves around the character). Triangles are cut only when they lie fully outside; small deviations stay; the holes a cut opens are closed; nothing is cut if it would take over half of the mesh;
- **Remove floating** deletes a detached piece only when it is proven junk: a crumb, or mostly outside the silhouettes. Small separate parts inside the silhouettes (clips, ribbons) stay.

Before and after the cleanup the log reports triangles, open edges and separate parts, and the live status shows the number of open edges.

## Neural start (Hunyuan3D 2.0)

Instead of a volume from the views, the Hunyuan3D 2.0 network (shape only, no texture) builds the start mesh from a colour image. All parts are kept and hidden places (for example under a skirt) are completed. Licence: Tencent Hunyuan Non-Commercial; read its terms before any commercial use (see `THIRD_PARTY.md`).

1. **Start mesh → Hunyuan3D → Install Hunyuan3D**. Sets up a separate environment with PyTorch and CUDA once, about 4 GB. Nothing is compiled.
2. **Model**: *mini* (light, default), *standard* (about 6 GB of video memory for the shape) or *multi-view* (three views: front, left at 90°, back at 180°; experimental). **Front colour image** is the front colour view; then press **Generate start mesh**. The first run downloads the weights, a few GB. *Detail grid* is the surface resolution, *Steps* the number of denoising steps.
3. **Fit to silhouettes** (a separate block, also available for the volume method) pushes or pulls the mesh so that its outline matches the silhouette of every view. Done once, then cached:
   - *Silhouette passes* — number of passes;
   - *Silhouette step* — share of the correction per pass;
   - *Move whole parts* — how far a correction spreads over the mesh: 0 moves only the rim, a high value moves a whole arm or the back;
   - *Silhouette smoothing* — smoothing after every pass;
   - *Silhouette resolution* — resolution of the silhouettes.
4. Then the usual fit to the normal maps, in the preview or in **Final build**.

On the test character the silhouette fit of a neural start mesh reduced the deviation from 2.4 % to 1.15 % of height (measured with an earlier start-mesh network that is no longer part of the add-on). The mesh generated by the Hunyuan3D 2.0 web demo scored 1.11 % on the same reference after a best-fit scale and rotation. The Hunyuan3D integration inside the add-on has not been measured yet.

## Baking maps from views (Bake maps from views)

Creates depth and normal maps from ordinary colour views, such as the 3D-style renders an AI draws following the companion skill (`view_000.png` … `view_315.png`). Runs on the GPU in the same environment as Hunyuan3D.

1. **Bake maps from views → Install map bakers.** Sets up the PyTorch CUDA environment once (if it does not exist yet) and the Depth Anything V2 and Marigold packages. The models themselves download on the first bake: about 1.3 GB for depth and about 2 GB for normals.
2. **Views folder** — the folder with the views. **Normals**: *Marigold* (separate network, better detail) or *From depth* (faster, softer). **Quality**: Fast / Normal / Best.
3. Optional checkboxes: **Convert to clay before baking** (repaints the colour views as flat light-grey clay first; reduces noise from colour and painted lighting) and **Remove drawn lighting** (de-lighting; not used together with clay).
4. **Bake maps.** For every view:
   - depth — Depth Anything V2;
   - normals — Marigold;
   - the normal axes are checked against the silhouette rim (on the left rim the normal points left, on the top rim up); the sign and scale of the depth come from the slope the normals define. No generator convention is taken on trust.
5. The add-on fills the views list by itself (`normal_XXX.png` + `depth_XXX.png` in `mvm_work/maps`; the number of views follows the files, with manual + / −) and keeps the method you selected. For method 3 **Auto depth offset** is on and **Auto depth scale** is off: the scale was set during baking.
6. Check `MVM_maps_sheet` in the Image Editor: colour | normals | depth for every view. Seen from the front the face is bluish, the right rim reddish, the top greenish; closer is lighter.
7. **Invert relief** (in the Views panel) flips the normals and depth of every view at once. Use it when the bake came out inside-out.

Check on a test character (real Depth Anything + Marigold, RTX 4070, 8 views in about 7 min):
- depth shape error 2.4–6 % of height, normals 13–21°; every view is guessed separately and the views do not converge on one shape;
- **method 1 on these maps — 0.95 % of height**, all parts present;
- method 3 on the same maps — 3.1 %: it trusts the depth and carves away thin parts (legs, arms, tail); with **Max carve depth** 3 % it reaches 1.08 %.

## Depth + normals (C2-C)

The most accurate mode when every view has a depth map. On the test character: exact maps give 0.21 % of height, normals 5°; maps with generator-like errors (relative depth, warp, blur) give 0.28 %, normals 10°.

1. **Start mesh → Depth + normals**.
2. In **Views** set a **Depth map** for every view, in the same frame as its normal map:
   - `.npy` — depth in scene units, larger = closer to the camera, NaN = background;
   - 16-bit PNG — 0 = background, mid-range = the vertical axis; **PNG depth range** is how many character heights the whole range covers; **Bright = far** if lighter means farther.
3. Settings:
   - **Grid preview** (96–256) / **Grid final** (192, 256, 384 or 512) — cells per character height; the final default is 256 (384 keeps strands and ribbons but takes about three times longer; 512 needs much more memory);
   - **Auto depth scale** — finds the depth scale of every view from its normal map (relative depth from a generator; depth jumps at occlusion edges are ignored). Turn it off if the depth is exact, in scene units, or comes from **Bake maps**, which already has the scale;
   - **Auto depth offset** — fits the depth shift of every view to the silhouettes;
   - **Max carve depth** — limit of carving under the silhouette hull: 3 % for baked maps (3.1 → 1.08 % on the test character), 0 for exact maps;
   - **Smoothing (off = C2-B)** — detail-preserving smoothing; turn it off for a surface without smoothing;
   - **Also fit to normal maps** — an extra fit to the normal maps, usually not needed.
4. The volume and silhouette sliders are not used in this mode.

## Clearing the cache

A red **Clear cache** button at the bottom of the panel. It shows a list with the size of every item before deleting.

- **Always deleted:**
  - downloaded models — Hunyuan3D, Depth Anything V2, Marigold — from the Hugging Face cache (`%USERPROFILE%\.cache\huggingface\hub`); other models in the cache are left alone;
  - the add-on's temporary files in the work folder: `aligned`, `aligned_preview`, `maps`, `*.npz`, `*.json`, `*.obj`. Foreign files in the work folder stay; an empty folder is removed.
- **"Also installed environments" checkbox:** additionally the PyTorch / Hunyuan3D / map-baker environment (`mvm_neural`, about 6 GB), the add-on dependencies (`mvm_deps`) and the pip download cache. Afterwards **Install dependencies** and **Install …** are needed again.

Meshes in the scene stay. The live preview is stopped. Map paths that pointed into deleted folders are cleared. The button does not work while a background job is running.

## Resources

Computed on the CPU with numpy and scipy; the GPU is not used for the fit. Reference figures from tests on 2 cores: 8 views of 1024 px, a 2.2-million-triangle mesh — about 2 minutes per iteration; 2048 px and about 5 million triangles — about 5 minutes per iteration, peak memory about 6 GB. Every iteration runs as a separate process. On a machine with more cores (for example 16) the per-view work runs in parallel threads. If memory runs short, choose **Maps (final) = 1024**, lower **Final res**, or raise the **Finer mesh above** limit.

## Limitations

- What no view sees is completed smoothly.
- Thin parts below the carve limit keep only their silhouette.
- The input views must agree with each other. The fit discards single disagreeing pixels but cannot repair a view that contradicts the others as a whole.

**Fit to normal maps** has its own checkbox in every method. Switch it off to keep the start mesh as it is (for example a clean Hunyuan3D mesh when the baked normal maps are noisy); the silhouette fit and the cleanup still apply.

**Hunyuan3D input.** The background of the colour image is removed before generation: from the alpha channel if the PNG has one, otherwise by flooding the smooth background from the image border (flat colour, gradient, vignette or a soft floor shadow all work; the white eyes and a cream skirt are kept because outlines stop the flood). A PNG with a transparent background from any remover is always the most reliable input.

**Limits on the fits.** Both fits can be bounded so that they cannot ruin a good start mesh:
- *Fit to silhouettes → Max shift, % height* (default 2): no vertex may end up farther than this from the start mesh. Views whose silhouette does not match the mesh (IoU more than 0.15 below the median) are ignored, and the log lists the IoU of every view.
- *Fit to normal maps → Max shift per pass, % height* (default 0.5): the largest move of any vertex in one pass, so the total is at most passes x this value.

The normal-map fit follows the maps, noise included. Maps baked from AI-drawn colour views are guesses of 13-21 degrees per view, so on a clean Hunyuan3D mesh the fit adds bumps rather than detail; switch it off there.
