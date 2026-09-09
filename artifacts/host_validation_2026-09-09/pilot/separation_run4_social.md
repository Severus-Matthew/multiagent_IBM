| scenario | control | score | clean | gate | dep | degraded | edges | logs | reason |
|---|---|---:|---:|---|---:|---:|---:|---:|---|
| gen_k8s_target_port_misconfig-detection-user-ser | positive | 1.0 | 0.4667 | True | 1.0 | 1.0 | 0.0 | 1.0 | positive_scoped_channel_overlap |
| gen_k8s_target_port_misconfig-detection-user-ser | wrong_service | 0.5667 | 0.4667 | True | 1.0 | 0.0 | 0.0 | 0.5 | positive_scoped_channel_overlap |
| gen_k8s_target_port_misconfig-detection-user-ser | wrong_mechanism | 0.7259 | 0.4667 | True | 0.931 | 1.0 | 0.0 | 1.0 | positive_scoped_channel_overlap |
| gen_scale_pod_zero_social_net-detection-user-ser | positive | 1.0 | 0.4345 | True | 1.0 | 1.0 | 0.0 | 1.0 | positive_scoped_channel_overlap |
| gen_scale_pod_zero_social_net-detection-user-ser | wrong_service | 0.5044 | 0.4345 | True | 0.8667 | 0.0 | 0.0 | 0.5 | positive_scoped_channel_overlap |
| gen_scale_pod_zero_social_net-detection-user-ser | wrong_mechanism | 0.4345 | 0.4345 | False | 0.931 | 0.0 | 0.0 | 0.0 | positive_scoped_channel_overlap |
