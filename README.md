# Komet

Hyesung Kim's blog, built on [AstroPaper](https://github.com/satnaing/astro-paper) (Astro).
Korean is the default language; English is available under `/en/`.

## Writing posts

Posts live in a language folder. The folder never appears in the URL.

| File                             | URL                 |
| -------------------------------- | ------------------- |
| `src/content/posts/ko/<slug>.md` | `/posts/<slug>/`    |
| `src/content/posts/en/<slug>.md` | `/en/posts/<slug>/` |

Frontmatter:

```yaml
---
title: Hello
description: One-line summary
pubDatetime: 2026-10-03T00:00:00+09:00
tags: [blog]
draft: false
# optional
category: Engineering # Engineering | Building | Decisions | Experiments | Reflections
type: build-log # see POST_TYPES in src/content.config.ts
project: my-project
translationKey: hello # same value on the ko and en versions links them in the KO | EN switch
---
```

## Commands

```sh
pnpm install
pnpm run dev      # http://localhost:4321
pnpm run build    # astro check + build + pagefind index
pnpm run lint
```

## Deploy

`.github/workflows/deploy.yml` builds and deploys to GitHub Pages on every push to `main`.
Site settings (title, URL, socials) are in `astro-paper.config.ts`.
