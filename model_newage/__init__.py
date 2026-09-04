"""Two architectures aimed at the one axis the measurements actually revealed.

Every single-path change we tried traded the same way: interventions that widen
or adapt the receptive field (SAM2's Hiera trunk, ADMSB's deformable
multi-scale block) gained on the texture-defined classes (crazing,
rolled-in_scale) and lost on the crisply bounded ones (patches, inclusion), so
the overall mAP never moved.  Both files here attack that zero-sum directly.

freqdual   -- two paths with a class-conditioned gate, so texture and boundary
              defects stop competing for the same weights
instdecomp -- predicts how many boxes a defect region should become, instead of
              leaving that to NMS
"""
