# Quality Score (0–100)

Start at 0 and add/subtract:

+20  salary present (min or max)
+15  apply_url present and valid http(s)
+20  description_html length ≥ 800 chars
+15  ≥ 5 hard skills identified
+10  normalized_title present
+10  seniority present
+10  remote_mode present
-20  spam_flag == true

Clamp to [0, 100].
