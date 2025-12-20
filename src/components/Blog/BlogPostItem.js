import React from 'react';
import Link from '@docusaurus/Link';
import {useBlogPost} from '@docusaurus/plugin-content-blog/client';
import ThemedImage from '@theme/ThemedImage';
import MDXContent from '@theme/MDXContent';

import styles from './BlogPostItem.module.css';

function BlogPostItem(props) {
  const {children, frontMatter, metadata, truncated, isBlogPostPage} = props;
  const {date, formattedDate, readingTime, title, tags, image: metaImage} = metadata;
  const {author, image} = frontMatter;
  const {permalink, editUrl} = metadata;

  // Function to add emojis based on blog tags
  const getTitleWithEmoji = (title, tags) => {
    if (tags && tags.length > 0) {
      const firstTag = tags[0].label.toLowerCase();

      if (firstTag.includes('ai') || firstTag.includes('intelligence')) {
        return `🤖 ${title}`;
      } else if (firstTag.includes('robot') || firstTag.includes('robotics')) {
        return `🤖 ${title}`;
      } else if (firstTag.includes('education') || firstTag.includes('learning')) {
        return `🎓 ${title}`;
      } else if (firstTag.includes('hackathon') || firstTag.includes('project')) {
        return `🏆 ${title}`;
      } else if (firstTag.includes('tech') || firstTag.includes('technology')) {
        return `💻 ${title}`;
      } else if (firstTag.includes('code') || firstTag.includes('programming')) {
        return `⌨️ ${title}`;
      } else if (firstTag.includes('future') || firstTag.includes('innovation')) {
        return `🔮 ${title}`;
      } else if (firstTag.includes('research') || firstTag.includes('study')) {
        return `🔬 ${title}`;
      } else {
        return `📝 ${title}`;
      }
    }
    return `📝 ${title}`;
  };

  const titleWithEmoji = getTitleWithEmoji(title, tags);

  const renderPostHeader = () => (
    <header>
      <h2 className="blog-post-title">
        <Link to={permalink} className="text--truncate blog-post-title-link">
          {titleWithEmoji}
        </Link>
        {readingTime && (
          <span className="blog-post-reading-time">
            {Math.ceil(readingTime)} min read
          </span>
        )}
      </h2>
      <div className="blog-post-meta">
        <time dateTime={date} className="blog-post-date">
          {formattedDate}
        </time>
      </div>
    </header>
  );

  return (
    <article className={`blog-post-item ${styles.blogPostItem}`}>
      {isBlogPostPage && (
        <header>
          <h1 className="blog-post-title">
            <Link to={permalink} className="text--truncate blog-post-title-link">
              {titleWithEmoji}
            </Link>
            {readingTime && (
              <span className="blog-post-reading-time">
                {Math.ceil(readingTime)} min read
              </span>
            )}
          </h1>
          <div className="blog-post-meta">
            <time dateTime={date} className="blog-post-date">
              {formattedDate}
            </time>
          </div>
        </header>
      )}
      {!isBlogPostPage && renderPostHeader()}

      {image && (
        <div className="margin-vert--md">
          <ThemedImage
            alt={title}
            sources={image}
            className={styles.image}
          />
        </div>
      )}

      <div className="markdown">
        <MDXContent>{children}</MDXContent>
      </div>

      {!isBlogPostPage && truncated && (
        <div className="margin-vert--md">
          <Link
            className="button button--outline button--primary"
            to={permalink}>
            Read More
          </Link>
        </div>
      )}

      {tags.length > 0 && (
        <div className="blog-tags">
          {tags.map(({label, permalink: tagPermalink}) => (
            <Link
              key={tagPermalink}
              className="blog-tag"
              to={tagPermalink}>
              #{label}
            </Link>
          ))}
        </div>
      )}

      {editUrl && !isBlogPostPage && (
        <div className="margin-vert--md">
          <Link to={editUrl}>Edit this post</Link>
        </div>
      )}
    </article>
  );
}

export default BlogPostItem;