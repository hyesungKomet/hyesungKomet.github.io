import { getRelativeLocaleUrl } from "astro:i18n";
import { BLOG_PATH } from "@/content.config";
import { slugifyStr } from "./slugify";
import { POST_LANGS, getPostLangFromId } from "./postLang";

function getPostPathSegments(filePath: string | undefined): string[] {
  const segments =
    filePath
      ?.replace(BLOG_PATH, "")
      .split("/")
      .filter(path => path !== "")
      .filter(path => !path.startsWith("_"))
      .slice(0, -1) ?? [];
  // The leading language folder (`ko/`, `en/`) never appears in URLs;
  // the language is expressed by the locale prefix instead.
  if (POST_LANGS.includes(segments[0] as (typeof POST_LANGS)[number])) {
    segments.shift();
  }
  return segments.map(segment => slugifyStr(segment));
}

function getIdSlug(id: string): string {
  const postId = id.split("/");
  return postId.length > 0 ? String(postId[postId.length - 1]) : id;
}

function getPostSlugPath(id: string, filePath: string | undefined): string {
  const pathSegments = getPostPathSegments(filePath);
  const slug = getIdSlug(id);
  return pathSegments.length > 0
    ? [...pathSegments, slug].join("/")
    : String(slug);
}

/**
 * Returns the slug-only path for use as a route param in `getStaticPaths`.
 * No base prefix, no locale — Astro handles those at a higher level.
 * The language folder is stripped: `ko/my-post.md` and `en/my-post.md` both
 * map to `/my-post`.
 */
export function getPostSlug(id: string, filePath: string | undefined): string {
  return `/${getPostSlugPath(id, filePath)}`;
}

/**
 * Returns a fully navigable URL for use in `<a href>` and RSS links.
 * Applies both locale routing and the configured Astro base via
 * `getRelativeLocaleUrl`. The locale defaults to the post's own language.
 * e.g. `/posts/my-post` (ko) or `/en/posts/my-post` (en)
 */
export function getPostUrl(
  id: string,
  filePath: string | undefined,
  locale: string = getPostLangFromId(id, filePath)
): string {
  return getRelativeLocaleUrl(locale, `posts/${getPostSlugPath(id, filePath)}`);
}
