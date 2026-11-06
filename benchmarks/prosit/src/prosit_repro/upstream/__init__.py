"""Vendored chemistry/tensorization primitives from kusterlab/prosit.

Source: https://github.com/kusterlab/prosit (Apache-2.0; see LICENSE in this dir).

We import these unchanged so that mass constants, the amino-acid alphabet,
ION_OFFSET, peptide_parser, get_forward_backward / get_annotation /
binarysearch, parse_ion, and the sanitize chain (cap / mask_outofrange /
mask_outofcharge / reshape_flat) match the original Prosit numerical path
bit-for-bit. Only relative imports inside the upstream files were touched
so the package still resolves under our path.

Do not edit these files. If a fix is needed, fork upstream and re-vendor.
"""
