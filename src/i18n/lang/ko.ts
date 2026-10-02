import type { UIStrings } from "../types";

export default {
  nav: {
    home: "홈",
    posts: "글",
    tags: "태그",
    about: "소개",
    archives: "아카이브",
    search: "검색",
  },
  post: {
    publishedAt: "게시일",
    updatedAt: "수정",
    sharePostIntro: "이 글 공유하기:",
    sharePostOn: "{{platform}}에 공유하기",
    sharePostViaEmail: "이메일로 공유하기",
    tagLabel: "태그",
    backToTop: "맨 위로",
    goBack: "돌아가기",
    editPage: "이 페이지 수정하기",
    previousPost: "이전 글",
    nextPost: "다음 글",
    category: "분류",
    type: "유형",
    project: "프로젝트",
  },
  pagination: {
    prev: "이전",
    next: "다음",
    page: "페이지",
  },
  home: {
    socialLinks: "링크",
    featured: "추천 글",
    recentPosts: "최근 글",
    allPosts: "전체 글 보기",
  },
  footer: {
    copyright: "Copyright",
    allRightsReserved: "All rights reserved.",
  },
  pages: {
    tagTitle: "태그",
    tagDesc: "다음 태그가 붙은 글 모음:",

    tagsTitle: "태그",
    tagsDesc: "글에 쓰인 모든 태그입니다.",

    postsTitle: "글",
    postsDesc: "지금까지 쓴 모든 글입니다.",

    archivesTitle: "아카이브",
    archivesDesc: "날짜별로 모아 본 모든 글입니다.",

    searchTitle: "검색",
    searchDesc: "글을 검색해 보세요.",
  },
  a11y: {
    skipToContent: "본문으로 건너뛰기",
    openMenu: "메뉴 열기",
    closeMenu: "메뉴 닫기",
    toggleTheme: "테마 전환",
    searchPlaceholder: "글 검색...",
    noResults: "검색 결과가 없습니다",
    goToPreviousPage: "이전 페이지로",
    goToNextPage: "다음 페이지로",
  },
  langSwitch: {
    label: "언어 선택",
  },
  notFound: {
    title: "404 페이지를 찾을 수 없음",
    message: "페이지를 찾을 수 없습니다",
    goHome: "홈으로 돌아가기",
  },
} satisfies UIStrings;
