interface MetadataProps {
  title: string;
  description?: string;
  ogTitle?: string;
  ogDescription?: string;
  ogUrl?: string;
  ogImage?: string;
  ogSiteName?: string;
  ogType?: string;
  twitterCard?: string;
  twitterSite?: string;
}

export default function Metadata({
  title,
  description,
  ogTitle,
  ogDescription,
  ogUrl,
  ogImage,
  ogSiteName,
  ogType,
  twitterCard,
  twitterSite,
}: MetadataProps) {
  return (
    <>
      <title>{title}</title>
      {description ? <meta content={description} name="description" /> : null}
      {ogTitle ? <meta content={ogTitle} property="og:title" /> : null}
      {ogDescription ? <meta content={ogDescription} property="og:description" /> : null}
      {ogUrl ? <meta content={ogUrl} property="og:url" /> : null}
      {ogImage ? <meta content={ogImage} property="og:image" /> : null}
      {ogSiteName ? <meta content={ogSiteName} property="og:site_name" /> : null}
      {ogType ? <meta content={ogType} property="og:type" /> : null}
      {twitterCard ? <meta content={twitterCard} name="twitter:card" /> : null}
      {twitterSite ? <meta content={twitterSite} name="twitter:site" /> : null}
    </>
  );
}