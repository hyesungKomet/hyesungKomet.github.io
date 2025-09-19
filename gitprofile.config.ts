// gitprofile.config.ts

const CONFIG = {
  github: {
    username: 'hyesungKomet', // Your GitHub org/user name. (This is the only required config)
    customAvatar: '/KHS.jpg', // Custom profile image URL
  },
  /**
   * If you are deploying to https://<USERNAME>.github.io/, for example your repository is at https://github.com/arifszn/arifszn.github.io, set base to '/'.
   * If you are deploying to https://<USERNAME>.github.io/<REPO_NAME>/,
   * for example your repository is at https://github.com/arifszn/portfolio, then set base to '/portfolio/'.
   */
  base: '/',
  projects: {
    github: {
      display: false, // Display GitHub projects?
      header: 'Github Projects',
      mode: 'automatic', // Mode can be: 'automatic' or 'manual'
      automatic: {
        sortBy: 'stars', // Sort projects by 'stars' or 'updated'
        limit: 8, // How many projects to display.
        exclude: {
          forks: false, // Forked projects will not be displayed if set to true.
          projects: [], // These projects will not be displayed. example: ['arifszn/my-project1', 'arifszn/my-project2']
        },
      },
      manual: {
        // Properties for manually specifying projects
        projects: ['arifszn/gitprofile', 'arifszn/pandora'], // List of repository names to display. example: ['arifszn/my-project1', 'arifszn/my-project2']
      },
    },
    about: {
      header: 'About Me',
      // To hide the `About Me` section, keep it empty.
      projects: [
        {
          title: '',
          description:
            'I am a **4th-year Computer Education student** at Sungkyunkwan University, pursuing a **double major in Applied Artificial Intelligence**. My primary research interests lie in **NLP (Natural Language Processing)** and **Multimodal AI**, where I enjoy exploring the intersection of different data modalities and their applications. I have a strong passion for **experiencing diverse fields and domains**, constantly seeking to broaden my knowledge and skills across various technological landscapes. In my free time, I enjoy playing **table tennis** and working out at the **gym** to maintain a healthy work-life balance.',
          imageUrl: '',
          link: '',
        },
      ],
    },
    external: {
      header: 'My Projects',
      // To hide the `External Projects` section, keep it empty.
      projects: [
        {
          title: 'Google Machine Learning Bootcamp 2024',
          description:
            'Jul 2024– Oct 2024 • Developed a Korean psychological counseling chatbot by fine-tuning the Gemma2-2B-IT model with PEFT techniques during the Gemma Sprint Project',
          imageUrl:
            '/jung.jpg',
          link: 'https://github.com/hyesungKomet/Gemma2-2B-IT_PEFT_JungBot',
        },
        {
          title: 'Wasabi Warriors– Artificial Intelligence Course',
          description:
            'May 2024- Jun 2024 • Collaborated with a team of three to perform multi-class classification and object detection on sushi images • Applied transfer learning techniques and evaluated model performance using standard metrics (accuracy, mAP)',
          imageUrl:
            '/wasabi_warriors.jpg',
          link: 'https://github.com/hyesungKomet/Wasabi_Warriors',
        },
      ],
    },
  },
  seo: { title: 'Portfolio of hyesungKomet', description: '', imageURL: '' },
  social: {
    linkedin: '',
    x: '',
    mastodon: '',
    researchGate: '',
    facebook: '',
    instagram: '',
    reddit: '',
    threads: '',
    youtube: '', // example: 'pewdiepie'
    udemy: '',
    dribbble: '',
    behance: '',
    medium: '',
    dev: '',
    stackoverflow: '', // example: '1/jeff-atwood'
    discord: '',
    telegram: '',
    website: 'https://hyesungKomet.github.io',
    phone: '+82 (0)10-7184-4372',
    email: 'khsfun0312khsfun@gmail.com',
  },
  resume: {
    fileUrl:
      '/Hyesung_Kim_CV.pdf', // Empty fileUrl will hide the `Download CV` button.
  },
  skills: [
    'Python',
    'PyTorch',
    'JavaScript',
    'C',
    'React Native',
    'MySQL',
    'Git',
    'Docker',
  ],
  experiences: [
    {
      company: 'HLILab',
      position: 'Intern',
      from: 'July 2025',
      to: 'Present',
      companyLink: '',
    },
    {
      company: 'Bundang High School',
      position: 'Teaching Practicum',
      from: 'July 2025',
      to: 'Present',
      companyLink: '',
    },
    {
      company: 'LG Aimers',
      position: '6th Cohort Participant',
      from: 'January 2025',
      to: 'February 2025',
      companyLink: '',
    },
    {
      company: 'Google Machine Learning Bootcamp',
      position: 'Participant',
      from: 'July 2024',
      to: 'October 2024',
      companyLink: '',
    },
    {
      company: 'Republic of Korea Air Force',
      position: 'System Development Unit',
      from: 'March 2022',
      to: 'December 2023',
      companyLink: '',
    },
  ],
  certifications: [
    {
      name: 'Engineer Information Processing (정보처리기사)',
      body: 'Korea Research Institute for Vocational Education & Training',
      year: 'June 2025',
      link: '',
    },
    {
      name: 'ADSP (Advanced Data Analytics Semi-Professional Certification) (데이터분석 준전문가)',
      body: 'Korea Data Agency',
      year: 'September 2024',
      link: '',
    },
    {
      name: 'Industrial Engineer Information Processing (정보처리산업기사)',
      body: 'Korea Research Institute for Vocational Education & Training',
      year: 'November 2023',
      link: '',
    },
    {
      name: 'Industrial Engineer Information Processing (정보기기운용기능사)',
      body: 'Korea Research Institute for Vocational Education & Training',
      year: 'November 2023',
      link: '',
    },
    {
      name: 'SQLD (SQL Developer Certification) (SQL 개발자)',
      body: 'Korea Data Agency',
      year: 'September 2023',
      link: '',
    },
  ],
  educations: [
    {
      institution: 'Sungkyunkwan University',
      degree: 'Bachelor of Computer Education',
      from: '2020',
      to: 'Present',
    },
  ],
  publications: [
    // Publications will be displayed here when available
  ],
  // Display articles from your medium or dev account. (Optional)
  blog: {
    source: 'dev', // medium | dev
    username: 'hyesungKomet', // to hide blog section, keep it empty
    limit: 2, // How many articles to display. Max is 10.
  },
  googleAnalytics: {
    id: '', // GA3 tracking id/GA4 tag id UA-XXXXXXXXX-X | G-XXXXXXXXXX
  },
  // Track visitor interaction and behavior. https://www.hotjar.com
  hotjar: { id: '', snippetVersion: 6 },
  themeConfig: {
    defaultTheme: 'lofi',

    // Hides the switch in the navbar
    // Useful if you want to support a single color mode
    disableSwitch: false,

    // Should use the prefers-color-scheme media-query,
    // using user system preferences, instead of the hardcoded defaultTheme
    respectPrefersColorScheme: false,

    // Display the ring in Profile picture
    displayAvatarRing: true,

    // Available themes. To remove any theme, exclude from here.
    themes: [
      'light',
      'dark',
      'cupcake',
      'bumblebee',
      'emerald',
      'corporate',
      'synthwave',
      'retro',
      'cyberpunk',
      'valentine',
      'halloween',
      'garden',
      'forest',
      'aqua',
      'lofi',
      'pastel',
      'fantasy',
      'wireframe',
      'black',
      'luxury',
      'dracula',
      'cmyk',
      'autumn',
      'business',
      'acid',
      'lemonade',
      'night',
      'coffee',
      'winter',
      'dim',
      'nord',
      'sunset',
      'caramellatte',
      'abyss',
      'silk',
      'procyon',
    ],
  },

  // Optional Footer. Supports plain text or HTML.
  footer: `Made with <a 
      class="text-primary" href="https://github.com/hyesungKomet/hyesungKomet.github.io"
      target="_blank"
      rel="noreferrer"
    >GitProfile</a> and ❤️`,

  enablePWA: true,
};

export default CONFIG;
