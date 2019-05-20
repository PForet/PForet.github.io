---
defaults:
  # _posts
  - scope:
      path: ""
      type: posts
    values:
      layout: single
      author_profile: true
      read_time: true
      comments: true
      share: true
      related: false
      use_math: true

excerpt: "Test page"
---

This is a test for video embedding.

In table:


| Dual of the original function | Dual of the dual |
|:--:|:--:|
|![convex conjugate]({{ "/assets/videos/convex/conjugate_clean_convex.mp4" | absolute_url }}) | ![convex conjugate]({{ "/assets/videos/convex/conjugate_clean_convex.mp4" | absolute_url }})|

Solo:

![convex conjugate]({{ "/assets/videos/convex/conjugate_clean_convex.mp4" | absolute_url }})