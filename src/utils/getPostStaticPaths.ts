import { getCollection } from "astro:content";
import { getSortedPosts } from "./getSortedPosts";
import { getPostSlug } from "./getPostPaths";
import { isLang, type PostLang } from "./postLang";

export type AdjacentPost = {
  id: string;
  title: string;
  filePath: string | undefined;
} | null;

/**
 * `getStaticPaths` result for post detail routes of one language.
 * Prev/next navigation stays within the same language.
 */
export async function getPostStaticPaths(lang: PostLang) {
  const posts = await getCollection("posts", isLang(lang));
  const sortedPosts = getSortedPosts(posts);

  const toAdjacent = (index: number): AdjacentPost => {
    const post = sortedPosts[index];
    return post
      ? { id: post.id, title: post.data.title, filePath: post.filePath }
      : null;
  };

  return sortedPosts.map((post, index) => ({
    params: { slug: getPostSlug(post.id, post.filePath) },
    props: {
      post,
      // sortedPosts is newest-first, so "older" (prev) is a higher index
      // and "newer" (next) is a lower index.
      prevPost: toAdjacent(index + 1),
      nextPost: index > 0 ? toAdjacent(index - 1) : null,
    },
  }));
}
