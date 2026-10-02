import type { CollectionEntry } from "astro:content";
import config from "@/config";
import { postFilter } from "./postFilter";

export const POST_LANGS = ["ko", "en"] as const;
export type PostLang = (typeof POST_LANGS)[number];

const DEFAULT_LANG = config.site.lang as PostLang;

function isPostLang(value: string | undefined): value is PostLang {
  return POST_LANGS.includes(value as PostLang);
}

/**
 * Returns a post's language from its leading folder,
 * e.g. `ko/hello` → "ko", `en/hello` → "en".
 * Posts outside a language folder fall back to the site default.
 */
export function getPostLangFromId(
  id: string,
  filePath?: string | undefined
): PostLang {
  const fromId = id.split("/")[0];
  if (isPostLang(fromId)) return fromId;
  const fromPath = filePath
    ?.replace(/^.*src\/content\/posts\//, "")
    .split("/")[0];
  if (isPostLang(fromPath)) return fromPath;
  return DEFAULT_LANG;
}

export function getPostLang(
  post: Pick<CollectionEntry<"posts">, "id" | "filePath">
): PostLang {
  return getPostLangFromId(post.id, post.filePath);
}

/** Filter predicate factory: `posts.filter(isLang("ko"))`. */
export function isLang(lang: string) {
  return (post: Pick<CollectionEntry<"posts">, "id" | "filePath">) =>
    getPostLang(post) === lang;
}

/** Finds the counterpart post (same `translationKey`) in another language. */
export function findTranslation(
  post: CollectionEntry<"posts">,
  posts: CollectionEntry<"posts">[],
  targetLang: PostLang
): CollectionEntry<"posts"> | undefined {
  const key = post.data.translationKey;
  if (!key) return undefined;
  return posts.find(
    p =>
      p.data.translationKey === key &&
      getPostLang(p) === targetLang &&
      postFilter(p)
  );
}
