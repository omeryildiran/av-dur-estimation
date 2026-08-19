\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{biorxiv}[2026/08/19 bioRxiv-compatible preprint class]

\newif\ifbiorxiv@linenumbers
\DeclareOption{linenumbers}{\biorxiv@linenumberstrue}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax

\LoadClass[11pt]{article}
% The manuscript uses PDF/PNG figures; avoid loading the optional EPS converter.
\providecommand{\DoNotLoadEpstopdf}{}
\RequirePackage{biorxiv}
\RequirePackage[margin=1in,headheight=15pt,headsep=18pt]{geometry}
\RequirePackage[round,authoryear]{natbib}
% Load float before hyperref so each figure receives only one PDF destination.
\RequirePackage{float}
\RequirePackage[colorlinks=true,linkcolor=biorxivblue,urlcolor=biorxivblue,
  citecolor=biorxivblue]{hyperref}
\RequirePackage{pifont}

\ifbiorxiv@linenumbers
  \IfFileExists{lineno.sty}{\RequirePackage{lineno}}{%
    \ClassWarning{biorxiv}{linenumbers requested, but lineno.sty was not found}%
    \biorxiv@linenumbersfalse}
\fi

\makeatletter
\newcommand{\biorxiv@title}{}
\newcommand{\biorxiv@abstract}{}
\newcommand{\biorxiv@keywords}{}
\newcommand{\biorxiv@authors}{}
\newcommand{\biorxiv@category}{Neuroscience}
\newcommand{\biorxiv@license}{CC BY 4.0}
\newcounter{biorxiv@authorcount}

\renewcommand{\title}[1]{\gdef\biorxiv@title{#1}}
\renewcommand{\abstract}[1]{\gdef\biorxiv@abstract{#1}}
\newcommand{\keywords}[1]{\gdef\biorxiv@keywords{#1}}
\newcommand{\biorxivcategory}[1]{\gdef\biorxiv@category{#1}}
\newcommand{\biorxivlicense}[1]{\gdef\biorxiv@license{#1}}

% Compatible with the six-field author command used by main_jov.tex:
% family, given names, institution, address, homepage, email.
\renewcommand{\author}[6]{%
  \stepcounter{biorxiv@authorcount}%
  \g@addto@macro\biorxiv@authors{%
    \par\textbf{#2 #1}\if\relax\detokenize{#6}\relax\else\,\href{mailto:#6}{\textsuperscript{\normalfont\texttt{*}}}\fi\\[-0.15em]
    {\small #3, #4}%
    \if\relax\detokenize{#5}\relax\else\ {\small\href{#5}{Author website}}\fi
    \par\vspace{0.45em}}}

\renewcommand{\maketitle}{%
  \hypersetup{pageanchor=false}%
  \begin{titlepage}
  \thispagestyle{empty}
  \noindent\colorbox{biorxivlight}{%
    \parbox{\dimexpr\textwidth-2\fboxsep\relax}{%
      \small\textbf{bioRxiv preprint}\hfill
      Category: \biorxiv@category\quad License: \biorxiv@license}}
  \vspace{1.8em}
  {\raggedright\LARGE\bfseries\color{biorxivblue}\biorxiv@title\par}
  \vspace{1.5em}
  {\raggedright\biorxiv@authors\par}
  \vspace{1em}
  {\raggedright\small\textsuperscript{*} Corresponding author\par}
  \vspace{1.8em}
  \noindent\begin{minipage}{\textwidth}
    \colorbox{biorxivlight}{%
      \parbox{\dimexpr\linewidth-2\fboxsep\relax}{%
        \textbf{Abstract}\par\vspace{0.45em}\biorxiv@abstract}}
  \end{minipage}\par
  \vspace{1em}
  \noindent\textbf{Keywords:} \biorxiv@keywords
  \vfill
  \noindent\small\color{biorxivgray}
  This manuscript is a preprint and has not been certified by peer review.
  \end{titlepage}
  \hypersetup{pageanchor=true}%
  \ifbiorxiv@linenumbers\linenumbers\fi}
\makeatother
