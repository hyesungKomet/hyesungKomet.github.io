import { defineCollection } from "astro:content";
import { z } from "astro/zod";
import { glob } from "astro/loaders";
import config from "@/config";

export const BLOG_PATH = "src/content/posts";

export const POST_CATEGORIES = [
  "Engineering",
  "Building",
  "Decisions",
  "Experiments",
  "Reflections",
] as const;

export const POST_TYPES = [
  "incident",
  "engineering-decision",
  "migration",
  "performance",
  "build-log",
  "failed-approach",
  "constraint-driven",
  "architecture",
  "debugging",
  "benchmark",
  "system-evolution",
  "designing-for-failure",
  "lessons-learned",
] as const;

const posts = defineCollection({
  loader: glob({ pattern: "**/[^_]*.{md,mdx}", base: `./${BLOG_PATH}` }),
  schema: ({ image }) =>
    z.object({
      author: z.string().default(config.site.author),
      pubDatetime: z.date(),
      modDatetime: z.date().optional().nullable(),
      title: z.string(),
      featured: z.boolean().optional(),
      draft: z.boolean().optional(),
      tags: z.array(z.string()).default(["others"]),
      ogImage: image().or(z.string()).optional(),
      description: z.string(),
      canonicalURL: z.string().optional(),
      hideEditPost: z.boolean().optional(),
      timezone: z.string().optional(),
      category: z.enum(POST_CATEGORIES).optional(),
      type: z.enum(POST_TYPES).optional(),
      project: z.string().optional(),
      /** Shared key linking a ko post and its en counterpart. */
      translationKey: z.string().optional(),
    }),
});

const pages = defineCollection({
  loader: glob({ pattern: "**/[^_]*.{md,mdx}", base: "./src/content/pages" }),
  schema: z.object({
    title: z.string(),
    description: z.string().optional(),
    ogImage: z.string().optional(),
    canonicalURL: z.string().optional(),
  }),
});

export const collections = { posts, pages };
