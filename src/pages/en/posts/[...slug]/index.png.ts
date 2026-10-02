import { getPostOgStaticPaths } from "@/utils/postOgImage";

export { GET } from "@/utils/postOgImage";

export async function getStaticPaths() {
  return getPostOgStaticPaths("en");
}
