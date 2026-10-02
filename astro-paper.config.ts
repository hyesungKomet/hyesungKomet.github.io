import { defineAstroPaperConfig } from "./src/types/config";

export default defineAstroPaperConfig({
  site: {
    url: "https://blog.habitus-lab.com/",
    title: "Komet",
    description:
      "낯선 사유와 기술의 파편을 주워 모아 기묘한 형체를 만들고, 그 안에서 피어난 실패와 집착, 우연한 발견을 기록합니다.",
    author: "Hyesung Kim",
    profile: "https://github.com/hyesungKomet",
    lang: "ko",
    timezone: "Asia/Seoul",
    dir: "ltr",
  },
  posts: {
    perPage: 4,
    perIndex: 4,
    scheduledPostMargin: 15 * 60 * 1000,
  },
  features: {
    lightAndDarkMode: true,
    dynamicOgImage: true,
    showArchives: true,
    showBackButton: true,
    editPost: { enabled: false },
    search: "pagefind",
  },
  socials: [
    { name: "github", url: "https://github.com/hyesungKomet" },
    { name: "mail", url: "mailto:khsfun0312khsfun@gmail.com" },
  ],
  shareLinks: [
    { name: "x", url: "https://x.com/intent/post?url=" },
    { name: "facebook", url: "https://www.facebook.com/sharer.php?u=" },
    { name: "mail", url: "mailto:?subject=See%20this%20post&body=" },
  ],
});
