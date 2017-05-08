use strict;
my $line_num = 0;
while(my $line = <STDIN>)
{
  ++$line_num;

  $line =~ s/[\r\n]//g;
  my @words;
  if ($line_num == 1)
  {
    @words = split /\s/, $line;
  }
  <STDIN>;
  my $last_pos = 0;
  while (my $line = <STDIN>)
  {
    $line =~ s/[\r\n]//g;
    if ($line =~ /^$/)
    {
      $line_num = 0;
      last;
    } 
    else 
    {
      if ($line =~ /^(.+?)\((.+?)-(\d+?)'*, (.+?)-(\d+?)'*\)$/)
      {
        my $current_pos = $5;

        if ($current_pos eq $last_pos)
        {
          next;
        }

        if (($current_pos - $last_pos) ne 1)
        {
          for my $pp ($last_pos + 1..$current_pos-1)
          {
            print $pp."\t.\tPOS\t.\tP\n";
          }
         
        }
        my $modify_pos = $3 - 1;
        print $5."\t".$4."\tPOS\t".$modify_pos."\t".$1."\n";
        $last_pos = $5;
      }
      else
      {
        print STDERR "Format error!\n";
        print STDERR $line."\n";
      }
    }
  }

  if ($last_pos ne scalar(@words))
  { 
    for my $pp ($last_pos + 1..scalar(@words))
    {
      print $pp."\t.\tPOS\t.\tP\n";
    }
  }
  
  print "\n";

}
