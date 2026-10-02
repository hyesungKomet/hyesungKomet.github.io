/**
 * The site font (Google Sans Code) has no Hangul glyphs, so satori renders
 * Korean text in OG images as tofu boxes. This fetches a Noto Sans KR subset
 * containing only the characters in `text` from Google Fonts, to be passed
 * to satori as a fallback font.
 *
 * Returns an empty list (= no fallback) when the text has no Hangul or the
 * request fails, so OG generation never breaks the build.
 */
type SatoriFont = {
  name: string;
  data: ArrayBuffer;
  weight: 400 | 700;
  style: "normal";
};

const FAMILY = "Noto Sans KR";
const HANGUL = /[ᄀ-ᇿ㄰-㆏가-힯]/;

async function fetchSubset(
  weight: 400 | 700,
  text: string
): Promise<SatoriFont | null> {
  const cssUrl =
    `https://fonts.googleapis.com/css2?family=${encodeURIComponent(FAMILY)}` +
    `:wght@${weight}&text=${encodeURIComponent(text)}`;
  const css = await fetch(cssUrl).then(res => (res.ok ? res.text() : ""));
  const fontUrl = css.match(
    /src: url\((.+?)\) format\('(?:opentype|truetype)'\)/
  )?.[1];
  if (!fontUrl) return null;
  const data = await fetch(fontUrl).then(res => res.arrayBuffer());
  return { name: FAMILY, data, weight, style: "normal" };
}

export async function loadKoreanFonts(text: string): Promise<SatoriFont[]> {
  if (!HANGUL.test(text)) return [];
  const uniqueChars = [...new Set(text)].join("");
  try {
    const fonts = await Promise.all([
      fetchSubset(400, uniqueChars),
      fetchSubset(700, uniqueChars),
    ]);
    return fonts.filter((font): font is SatoriFont => font !== null);
  } catch {
    return [];
  }
}
