Jekyll::Hooks.register :posts, :pre_render do |post|
    slug = post.basename_without_ext.sub(/^\d{4}-\d{2}-\d{2}-/, "")
    post.data['media_subpath'] ||= "/posts/#{slug}"
  end